import os

import numpy as np
import torch
# import pycolmap

from matching import match_images
from utils.basic import Print
from utils.to_pycolmap import batch_torch_matrix_to_pycolmap
from utils.geometry import compute_epipolar_errors, homo
from tqdm import tqdm


def _visible_mean(values, mask):
    mask_f = mask.float()
    denom = mask_f.sum(0).clamp_min(1.0)
    return (values.float() * mask_f).sum(0) / denom


def _inv1p_score(values):
    return 1.0 / (1.0 + torch.clamp(values.float(), min=0.0))


def run_sfm(
        images,
        ff_outputs,
        match_models,
        cfg,
        gt=None,
        output_dir=None,
):
    import pycolmap
    images_ff = ff_outputs['images_ff']  # B,H,W,3
    N = images_ff.shape[0]
    device = images_ff.device
    ff_h, ff_w = images_ff.shape[1:3]
    output_dict = {} 

    """
    1. Dense matching
    """
    match_on_ff_res = os.environ.get("GGPT_MATCH_ON_FF_RES", "0").strip().lower() in {"1", "true", "yes", "y", "on"}
    if match_on_ff_res:
        images_for_matching = images_ff
        Print("GGPT_MATCH_ON_FF_RES=1: running matcher on feedforward-resolution images.")
    else:
        if isinstance(images, torch.Tensor)==False:
            # first convert to torch tensor
            images = torch.from_numpy(np.stack([np.array(img) for img in images], axis=0)).float()/255.0  #(N,H,W,3)
            images = images.to(device)
        images_for_matching = images
    m_h, m_w = images_for_matching.shape[1:3]
    sx, sy = ff_w/m_w, ff_h/m_h
    mres_to_fres = torch.tensor([[sx,0,0.5*(sx-1)],[0,sy,0.5*(sy-1)],[0,0,1]], dtype=torch.float32).to(device)  #3,3
    match_results = match_images(
        match_models=match_models,
        images_hr=images_for_matching.permute(0,3,1,2),  #N,3,H,W
        lr_h=ff_h, lr_w=ff_w,
        hr_to_lr=mres_to_fres
    )
    pred_scores = match_results['pred_scores']
    pred_cycle_error = match_results['pred_cycle_error']
    pred_scores_flat = pred_scores.reshape(N, -1)
    pred_cycle_error_flat = pred_cycle_error.reshape(N, -1)

    M_ba = (pred_scores > cfg.ba_config.score_thresh) & (pred_cycle_error < cfg.ba_config.cycle_err_thresh)
    M_dlt = (pred_scores > cfg.dlt_config.score_thresh) & (pred_cycle_error < cfg.dlt_config.cycle_err_thresh)

    """
    2. Sparse ba (TODO: support known camera poses)
    """
    # 2.1 Chooes tracks for BA
    M_ba = M_ba.reshape(N,-1) #Ntgt, Nsrc*H*W
    query_sp_scores = match_results['sp_scores'].view(-1)  #Nsrc*H*W
    tracknum_perview = torch.zeros(N, dtype=torch.int32).to(device)
    selected = torch.zeros(M_ba.shape[1], dtype=torch.bool).to(device) #Num_tracks=Nsrc*H*W
    assert M_ba.shape[1] == (N*ff_h*ff_w)
    for ni in tqdm(range(N), desc="Selecting tracks for BA"):
        to_select_num = cfg.ba_config.mintrack_per_view - tracknum_perview[ni]
        if to_select_num<=0:
            continue
        candidate_tracks = M_ba[ni]&(M_ba.sum(axis=0)>=2)&(~selected) # visible in this view, at least visible in 2 views, not selected, (Ntracks,)
        candidate_ids = torch.where(candidate_tracks)[0]
        if len(candidate_ids)==0:
            continue
        candidate_sps = query_sp_scores[candidate_ids]
        selected_ids = candidate_ids[torch.argsort(candidate_sps, descending=True)][:to_select_num]
        selected[selected_ids] = True
        tracknum_perview += M_ba[:,selected_ids].sum(axis=1).to(torch.int32)
    Print(f"The number of tracks covering each view: {[nn.item() for nn in tracknum_perview]}")
    tracks_ba = match_results['pred_matches_lr'].view(N,-1,2)[:,selected]  #N, Ntracks_ba, 2
    tracks_mask_ba = M_ba[:,selected]  #Ntgt, Ntracks_ba
    tracks_score_ba = pred_scores_flat[:,selected]
    tracks_cycle_ba = pred_cycle_error_flat[:,selected]
    if tracks_mask_ba.shape[1] == 0:
        raise AssertionError(
            "No BA tracks selected after score/cycle filtering; "
            f"views={N}, ff_hw=({ff_h},{ff_w}), "
            f"tracks_visible_ge_2={(M_ba.sum(axis=0) >= 2).sum().item()}, "
            f"mintrack_per_view={cfg.ba_config.mintrack_per_view}"
        )
    assert tracks_mask_ba.sum(axis=0).min()>=2, "Each track should be visible in at least two views for BA."
    pts3d_ba = ff_outputs['points'].reshape(-1,3)[selected]  #Ntracks_ba, 3 (Takes the ff's prediction at query positions as initialization)

    if cfg.match_config.save_vis:
        from matching.vis_match import vis_matches
        vis_matches(images=images_ff.permute(0,3,1,2),  #N,3,H,W 
            matches=tracks_ba,
            visibility=tracks_mask_ba, # visualize the matches with queries from image i
            filename=os.path.join(output_dir, f'matches_for_ba.png'), 
            vis_mask=tracks_mask_ba.sum(axis=0)>=2, # visualize len>=2 tracks (Ntgt,Num_tracks) (Actually it is all zero)
            vis_num_track=10)

    calibrated = cfg.ba_config.get('calibrated', False)
    ba_intrinsics = gt['intrinsics'][:,:3,:3].to(device).float() if calibrated else ff_outputs['intrinsics'][:,:3,:3]
    reconstruction = batch_torch_matrix_to_pycolmap(
        points3d = pts3d_ba,
        tracks = tracks_ba+0.5, masks = tracks_mask_ba,
        extrinsics = ff_outputs['extrinsics'][:,:3,:4],
        intrinsics = ba_intrinsics+0.5,
        image_size = [ff_w, ff_h],
        camera_type = cfg.ba_config.camera_type,
        shared_camera = cfg.ba_config.shared_camera,
    )
    refine_focal = not calibrated and cfg.ba_config.get('refine_focal_length', True)
    # refine_pp = not calibrated
    if cfg.ba_config.loss_function_type == 'cauchy':
        loss_function_type = pycolmap.LossFunctionType.CAUCHY
    else:
        loss_function_type = pycolmap.LossFunctionType.TRIVIAL
    ba_options = pycolmap.BundleAdjustmentOptions(
        loss_function_type=loss_function_type, loss_function_scale=cfg.ba_config.loss_function_scale,
        refine_focal_length=refine_focal)
    ba_options.solver_options.minimizer_progress_to_stdout = bool(int(os.environ.get("GGPT_BA_PROGRESS", "0")))
    ba_config = pycolmap.BundleAdjustmentConfig()
    for img_id in reconstruction.images.keys():
        ba_config.add_image(img_id)
    bundle_adjuster = pycolmap.create_default_bundle_adjuster(ba_options, ba_config, reconstruction)
    bundle_adjuster.solve()
    reconstruction.update_point_3d_errors()

    output_dict['intrinsics'] = torch.zeros_like(ff_outputs['intrinsics'])
    output_dict['extrinsics'] = torch.zeros_like(ff_outputs['extrinsics'])
    exp_cx_colmap = ff_w / 2.0
    exp_cy_colmap = ff_h / 2.0
    colmap_pp_tol = 1e-3
    for i in range(1,N+1): # Image ids in pycolmap start from 1
        camera = reconstruction.cameras[reconstruction.images[i].camera_id]
        cam_params = camera.params
        if camera.model == pycolmap.CameraModelId.SIMPLE_PINHOLE:
            fx = float(cam_params[0])
            fy = fx
            cx_colmap = float(cam_params[1])
            cy_colmap = float(cam_params[2])
        elif camera.model == pycolmap.CameraModelId.PINHOLE:
            fx = float(cam_params[0])
            fy = float(cam_params[1])
            cx_colmap = float(cam_params[2])
            cy_colmap = float(cam_params[3])
        else:
            raise AssertionError(f"Unsupported GGPT BA camera model: {camera.model}, params={cam_params}")
        if abs(cx_colmap - exp_cx_colmap) > colmap_pp_tol or abs(cy_colmap - exp_cy_colmap) > colmap_pp_tol:
            raise AssertionError(
                "GGPT BA returned a non-centered principal point, but this export path assumes fixed centered PP. "
                f"camera_params={cam_params}, expected_colmap_pp=({exp_cx_colmap:.6f},{exp_cy_colmap:.6f}), "
                f"got=({cx_colmap:.6f},{cy_colmap:.6f})"
            )
        output_dict['intrinsics'][i-1, 0, 0] = fx
        output_dict['intrinsics'][i-1, 1, 1] = fy
        output_dict['intrinsics'][i-1, 0, 2] = cx_colmap - 0.5
        output_dict['intrinsics'][i-1, 1, 2] = cy_colmap - 0.5
        output_dict['intrinsics'][i-1, 2, 2] = 1.0
        rigid3d = reconstruction.images[i].cam_from_world.matrix() # (3,4)
        output_dict['extrinsics'][i-1,:3,:] = torch.from_numpy(rigid3d).to(device).float()
        if output_dict['extrinsics'].shape[1] == 4:
            output_dict['extrinsics'][i-1,3,3] = 1.0
    output_dict['camera_success'] = True

    sfm_points_world = []
    sparse_score_variants = {"match_score_mean": [], "cycle_inv1p": [], "reproj_inv1p": []}
    sparse_intr = output_dict['intrinsics'][:, :3, :3].to(device).float()
    sparse_extr = output_dict['extrinsics'][:, :3, :4].to(device).float()
    tracks_ba_obs = tracks_ba.to(device).float() + 0.5
    match_score_mean_ba = _visible_mean(tracks_score_ba, tracks_mask_ba)
    cycle_score_ba = _inv1p_score(_visible_mean(tracks_cycle_ba, tracks_mask_ba))
    for pid in sorted(reconstruction.points3D.keys()):
        track_idx = int(pid) - 1
        if track_idx < 0 or track_idx >= tracks_ba.shape[1]:
            raise AssertionError(
                f"Unexpected BA point3D id={pid} for {tracks_ba.shape[1]} selected tracks."
            )
        p3d = reconstruction.points3D[pid]
        xyz = np.asarray(p3d.xyz, dtype=np.float32)
        if np.isfinite(xyz).all() is False:
            continue
        xyz_t = torch.from_numpy(xyz).to(device=device, dtype=torch.float32)
        xyz_h = torch.cat([xyz_t, xyz_t.new_tensor([1.0])], dim=0)
        pts_cam = torch.einsum('nij,j->ni', sparse_extr, xyz_h)
        uv_h = torch.einsum('nij,nj->ni', sparse_intr, pts_cam)
        uv = uv_h[:, :2] / uv_h[:, 2:3].clamp_min(1e-8)
        reproj_valid = tracks_mask_ba[:, track_idx] & torch.isfinite(uv).all(dim=1) & (pts_cam[:, 2] > 1e-8)
        if bool(reproj_valid.any()):
            reproj_err = torch.norm(uv[reproj_valid] - tracks_ba_obs[reproj_valid, track_idx], dim=-1).mean()
            reproj_inv1p = float(_inv1p_score(reproj_err))
        else:
            reproj_inv1p = 0.0
        match_score_mean = float(match_score_mean_ba[track_idx])
        cycle_inv1p = float(cycle_score_ba[track_idx])
        sfm_points_world.append(xyz)
        sparse_score_variants["match_score_mean"].append(match_score_mean)
        sparse_score_variants["cycle_inv1p"].append(cycle_inv1p)
        sparse_score_variants["reproj_inv1p"].append(reproj_inv1p)
    if len(sfm_points_world) > 0:
        output_dict["sfm_points_world"] = torch.from_numpy(np.asarray(sfm_points_world, dtype=np.float32)).to(device)
        output_dict["sfm_points_score"] = torch.from_numpy(
            np.asarray(sparse_score_variants["reproj_inv1p"], dtype=np.float32)
        ).to(device)
        output_dict["sfm_points_score_variants"] = {
            key: torch.from_numpy(np.asarray(values, dtype=np.float32)).to(device)
            for key, values in sparse_score_variants.items()
        }
    else:
        output_dict["sfm_points_world"] = torch.zeros((0, 3), dtype=torch.float32, device=device)
        output_dict["sfm_points_score"] = torch.zeros((0,), dtype=torch.float32, device=device)
        output_dict["sfm_points_score_variants"] = {
            "match_score_mean": torch.zeros((0,), dtype=torch.float32, device=device),
            "cycle_inv1p": torch.zeros((0,), dtype=torch.float32, device=device),
            "reproj_inv1p": torch.zeros((0,), dtype=torch.float32, device=device),
        }


    """
    3. Direct linear Triangulation
    """
    intrinsic_dlt, extrinsic_dlt = output_dict['intrinsics'].to(device), output_dict['extrinsics'].to(device) 
    P = torch.einsum('nij,njk->nik', intrinsic_dlt, extrinsic_dlt[:,:3,:4]).float() #N,3,4
    camR, camT = extrinsic_dlt[:,:3,:3], extrinsic_dlt[:,:3,-1]
    camC = -torch.einsum('nij,nj->ni', camR.permute(0,2,1), camT) #(N,3)
    # filter matches with epipolar error (TODO to make it more efficient)
    w2c_s, K_s = output_dict['extrinsics'].to(device), output_dict['intrinsics'].to(device)
    M_dlt = M_dlt.to(device)
    dlt_num = (M_dlt.sum(axis=0)>=2).sum().item()

    M_ba_debug = M_ba.reshape(N,N,-1).clone()
    for ni in range(N):
        w2c_0, K_0 = output_dict['extrinsics'][ni], output_dict['intrinsics'][ni]
        matches = match_results['pred_matches_lr'][:,ni].view(N,ff_h,ff_w,2) #(N, H,W, 2)
        dis_a, dis_b = compute_epipolar_errors(w2c_0, w2c_s, K_0, K_s, matches.view(N,ff_h,ff_w,2)) #(N,H,W)
        epipolar_errors_msk = (dis_a < cfg.dlt_config.max_epipolar_error) & (dis_b < cfg.dlt_config.max_epipolar_error) 
        epipolar_errors_msk[ni,:,:] = True  #self-view
        epipolar_errors_msk = epipolar_errors_msk.view(N, ff_h*ff_w) #(Ntgt, H*W)
        M_dlt[:,ni] = M_dlt[:,ni] & epipolar_errors_msk  #(Ntgt, H*W)
        M_ba_debug[:,ni] = M_ba_debug[:,ni] & epipolar_errors_msk  #(Ntgt, H*W)
    M_dlt = M_dlt.reshape(N,-1) #Ntgt, Nsrc*H*W
    remaining_tracks = (M_dlt.sum(axis=0)>=2)  #(Nsrc*H*W,)
    print(f"Number of tracks used for DLT triangulation: {remaining_tracks.sum().item()}")
    print("Epipolar error discard ratio: ", 1-(M_dlt.sum(axis=0)>=2).sum().item()/dlt_num)
    tracks_dlt = match_results['pred_matches_lr'].view(N,-1,2)[:,remaining_tracks].to(device)  #Nview, Ntracks_dlt, 2
    tracks_mask_dlt = M_dlt[:,remaining_tracks]  #Nview, Ntracks_dlt
    tracks_score_dlt = pred_scores_flat[:,remaining_tracks].to(device)
    tracks_cycle_dlt = pred_cycle_error_flat[:,remaining_tracks].to(device)

    weights = tracks_mask_dlt.float()
    max_pts_num = cfg.dlt_config.get('batch_size', 500000)
    num_chunk = (tracks_dlt.shape[1]+max_pts_num-1)//max_pts_num
    xyz_in_img, count_in_img = torch.zeros(N*ff_h*ff_w, 3).to(device), torch.zeros(N*ff_h*ff_w).to(device) # Our target
    dlt_score_sum = {
        "match_score_mean": torch.zeros(N*ff_h*ff_w, dtype=torch.float32, device=device),
        "cycle_inv1p": torch.zeros(N*ff_h*ff_w, dtype=torch.float32, device=device),
        "reproj_inv1p": torch.zeros(N*ff_h*ff_w, dtype=torch.float32, device=device),
    }
    for chunk_id in tqdm(range(num_chunk), desc='DLT triangulation'):
        start_idx = chunk_id*max_pts_num
        end_idx = min((chunk_id+1)*max_pts_num, tracks_dlt.shape[1])
        tracks_dlt_chunk = tracks_dlt[:,start_idx:end_idx]  #(Nview, Ntracks_chunk, 2)
        tracks_mask_dlt_chunk = tracks_mask_dlt[:,start_idx:end_idx]  #(Nview, Ntracks_chunk)
        tracks_score_dlt_chunk = tracks_score_dlt[:,start_idx:end_idx]
        tracks_cycle_dlt_chunk = tracks_cycle_dlt[:,start_idx:end_idx]
        
        Ai_chunk = tracks_dlt_chunk[...,None] * P[:,None,2:3,:] - P[:,None,:2,:] #(Nview,Ntracks,2,4)
        weights_chunk = weights[:,start_idx:end_idx] #(Nview, Ntracks_chunk)
        AitAi = Ai_chunk.permute(0,1,3,2) @ Ai_chunk #(Nview,Ntracks_chunk,4,4) 
        AitAi = AitAi * weights_chunk[:,:,None,None] #(Nview,Ntracks_chunk,4,4)
        AitAi_sum = AitAi.sum(axis=0) #(Ntracks_chunk,4,4)
        _, eigenvectors_chunk = torch.linalg.eigh(AitAi_sum) #(Ntracks_chunk,4), (Ntracks_chunk,4,4)
        pt3d_chunk = eigenvectors_chunk[:,:,0] #(Ntracks_chunk,4) Ascending order, the first column vector
        
        # Filter points with invalid solutions
        xyz = pt3d_chunk[:,:3] / pt3d_chunk[:,3:4] #(Npts,3)
        filter1 = pt3d_chunk[:,3].abs()>1e-10  #valid solution
        xyz = xyz[filter1]
        tracks_dlt_chunk = tracks_dlt_chunk[:,filter1]
        tracks_mask_dlt_chunk = tracks_mask_dlt_chunk[:,filter1]
        tracks_score_dlt_chunk = tracks_score_dlt_chunk[:,filter1]
        tracks_cycle_dlt_chunk = tracks_cycle_dlt_chunk[:,filter1]

        # Filter points based on reprojection error
        xy_reproj_chunk = torch.einsum('nij,mj->nmi', P, homo(xyz))
        xy_reproj_chunk = xy_reproj_chunk[:,:,:2]/xy_reproj_chunk[:,:,2:3]
        reproj_error_chunk = torch.norm(xy_reproj_chunk-tracks_dlt_chunk, dim=-1)
        reproj_error_mean_chunk = (reproj_error_chunk*tracks_mask_dlt_chunk).sum(0)/tracks_mask_dlt_chunk.sum(0)  
        filter_reproj = (reproj_error_mean_chunk < cfg.dlt_config.max_reproj_error)
        xyz = xyz[filter_reproj]
        tracks_dlt_chunk = tracks_dlt_chunk[:,filter_reproj]
        tracks_mask_dlt_chunk = tracks_mask_dlt_chunk[:,filter_reproj]
        tracks_score_dlt_chunk = tracks_score_dlt_chunk[:,filter_reproj]
        tracks_cycle_dlt_chunk = tracks_cycle_dlt_chunk[:,filter_reproj]
        reproj_error_mean_chunk = reproj_error_mean_chunk[filter_reproj]
        if filter_reproj.sum()==0:
            continue
        # Filter points based on triangulation angles
        # We need further batch operations here. quadratic to the number of views
        max_pts_num2 = int(min(max_pts_num//N, tracks_dlt_chunk.shape[1]))
        num_chunk2 = (tracks_dlt_chunk.shape[1]+max_pts_num2-1)//max_pts_num2
        for chunk_id2 in range(num_chunk2):
            start_idx2 = chunk_id2*max_pts_num2 
            end_idx2 = min((chunk_id2+1)*max_pts_num2, tracks_dlt_chunk.shape[1])
            rays_chunk = xyz[None,start_idx2:end_idx2,...] - camC[:,None,:] #(N,Ntracks_chunk,3)
            rays_chunk = rays_chunk / torch.linalg.norm(rays_chunk, axis=-1, keepdim=True) #(N,Ntracks_chunk,3)
            cos_angles = (rays_chunk[None,:,:,:]*rays_chunk[:,None,:,:]).sum(-1) #(N,N,Ntracks_chunk)-> (Nview1, Nview2, Ntracks_chunk)
            angle_radians = torch.acos(torch.clamp(cos_angles, -0.9999, 0.9999))
            angle_degs = torch.rad2deg(angle_radians)
            vismask_pairwise = tracks_mask_dlt_chunk[None,:, start_idx2:end_idx2] & tracks_mask_dlt_chunk[:,None,start_idx2:end_idx2]  #(Nview1, Nview2, Ntracks_chunk)
            angle_degs = angle_degs * vismask_pairwise.float() # Set invisible views to 0 angle 
            max_angle_degs_chunk = angle_degs.view(-1, end_idx2-start_idx2).max(0).values #(Ntracks_chunk,)
            if chunk_id2==0:
                max_angle_degs = max_angle_degs_chunk
            else:
                max_angle_degs = torch.cat([max_angle_degs, max_angle_degs_chunk], 0)
        filter_angle = (max_angle_degs > cfg.dlt_config.min_tri_angle)
        xyz = xyz[filter_angle]
        tracks_dlt_chunk = tracks_dlt_chunk[:,filter_angle]
        tracks_mask_dlt_chunk = tracks_mask_dlt_chunk[:,filter_angle]
        tracks_score_dlt_chunk = tracks_score_dlt_chunk[:,filter_angle]
        tracks_cycle_dlt_chunk = tracks_cycle_dlt_chunk[:,filter_angle]
        reproj_error_mean_chunk = reproj_error_mean_chunk[filter_angle]
        max_angle_degs = max_angle_degs[filter_angle]

        match_score_mean_chunk = _visible_mean(tracks_score_dlt_chunk, tracks_mask_dlt_chunk)
        cycle_score_chunk = _inv1p_score(_visible_mean(tracks_cycle_dlt_chunk, tracks_mask_dlt_chunk))
        reproj_inv1p_chunk = _inv1p_score(reproj_error_mean_chunk)

        xyz_for_img_chunk = xyz.unsqueeze(0).tile(N,1,1)[tracks_mask_dlt_chunk] #(N,Npts_chunk,3) -> (Nobs_chunk, 3)
        view_index_chunk = torch.arange(N).to(device).unsqueeze(1).tile(1,xyz.shape[0])[tracks_mask_dlt_chunk]  #(N,Npts) -> (Nobs_chunk,)
        index2d_obs_chunk = tracks_dlt_chunk[tracks_mask_dlt_chunk]
        index2d_in_img_chunk = index2d_obs_chunk.round().long()
        valid_obs = torch.isfinite(index2d_obs_chunk).all(dim=1)
        valid_obs = valid_obs & (index2d_in_img_chunk[:, 0] >= 0) & (index2d_in_img_chunk[:, 0] < ff_w)
        valid_obs = valid_obs & (index2d_in_img_chunk[:, 1] >= 0) & (index2d_in_img_chunk[:, 1] < ff_h)
        if valid_obs.sum() == 0:
            continue
        xyz_for_img_chunk = xyz_for_img_chunk[valid_obs]
        view_index_chunk = view_index_chunk[valid_obs]
        index2d_in_img_chunk = index2d_in_img_chunk[valid_obs]
        index1d_in_scene_chunk = view_index_chunk*ff_w*ff_h + index2d_in_img_chunk[:,1]*ff_w + index2d_in_img_chunk[:,0]  #(Nobs_chunk,)
        xyz_in_img.scatter_reduce_(dim=0, index=index1d_in_scene_chunk.unsqueeze(1).expand(-1,3), src=xyz_for_img_chunk, reduce='sum', include_self=True)
        count_in_img.scatter_reduce_(dim=0, index=index1d_in_scene_chunk, src=torch.ones_like(index1d_in_scene_chunk).float(), reduce='sum', include_self=True)
        track_scores_for_obs = {
            "match_score_mean": match_score_mean_chunk.unsqueeze(0).expand(N, -1)[tracks_mask_dlt_chunk][valid_obs],
            "cycle_inv1p": cycle_score_chunk.unsqueeze(0).expand(N, -1)[tracks_mask_dlt_chunk][valid_obs],
            "reproj_inv1p": reproj_inv1p_chunk.unsqueeze(0).expand(N, -1)[tracks_mask_dlt_chunk][valid_obs],
        }
        for key, obs_score in track_scores_for_obs.items():
            dlt_score_sum[key].scatter_reduce_(
                dim=0,
                index=index1d_in_scene_chunk,
                src=obs_score.float(),
                reduce='sum',
                include_self=True,
            )
    
    xyz_in_img = xyz_in_img / count_in_img.clamp(min=1)[:,None]
    dlt_xyz, count_in_img = xyz_in_img.view(N,ff_h,ff_w,3), count_in_img.view(N,ff_h,ff_w)
    dlt_mask = (count_in_img>0)  #(N,H,W)

    if dlt_mask.sum()==0:
        print('No valid DLT points after assigning to image pixels')
        output_dict['points_success'] = False
        return output_dict

    output_dict['points'] = dlt_xyz
    output_dict['point_masks'] = dlt_mask
    output_dict['point_scores'] = (dlt_score_sum["reproj_inv1p"] / count_in_img.view(-1).clamp(min=1.0)).view(N,ff_h,ff_w)
    output_dict['point_score_variants'] = {
        key: (value / count_in_img.view(-1).clamp(min=1.0)).view(N,ff_h,ff_w)
        for key, value in dlt_score_sum.items()
    }
    output_dict['points_success'] = True

    return output_dict
    
