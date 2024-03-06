import numpy as np

from h5py import File as hf
from pathlib import Path as Path

from plotly import graph_objects as go
from plotly.subplots import make_subplots


class TNGData:
    def __init__(self, dataPath):
        """
        dataPath == basePath
        
        """
        self.dataPath = dataPath
        
        # To access simulation.hdf5 file
        self.sim = sim = hf(self.dataPath / "simulation.hdf5", "r")
        
        # For book-keeping
        # Header has partMass, boxSize and Omega values
        self.header = header = dict(sim['Header'].attrs.items())
        self.partMass = header["MassTable"][1]
        
        self.numParticles = sim["Snapshots/99/PartType1/ParticleIDs"].shape[0]
        self.numHalos = sim["/Groups/99/Group/GroupLenType"].shape[0]
        self.numSubhalos = sim["/Groups/99/Subhalo/SubhaloLenType"].shape[0]

        self.P = np.asarray(sim["Snapshots/99/PartType1/Coordinates"])
        self.V = np.asarray(sim["Snapshots/99/PartType1/Velocities"])
        self.U = np.asarray(sim["Snapshots/99/PartType1/Potential"])
        self.halo_offsets = np.asarray(sim['Offsets/99/Group/SnapByType'])[:, 1]
        self.subhalo_offsets = np.asarray(sim['Offsets/99/Subhalo/SnapByType'])[:, 1]
        
    def __str__(self):
        return f"""TNGData - {self.dataPath.resolve()}
    num_part = {self.numParticles},
    num_halos = {self.numHalos},
    num_subhalos = {self.numSubhalos},
    part_mass (same for all) = {self.partMass},
    """

    def __repr__(self):
        return self.__str__()
    
    def get_halo_info(self, hid, raw=False):
        """
        Returns some details about the halo

        """
        halo = self.load_raw_halo(hid) if raw else self.load_halo(hid)        
        numpart_halo = halo.shape[0]

        n_sh_in_h = self.num_sh_in_h(hid)
        shids_hid = self.shids_in_hid(hid)
        np_sh_in_h = self.numpart_sh_in_h(hid)
        tar, (masked, numpart_true_bg) = self.prep_tar(hid)
        # numpart_true_bg -> particles bound only to the halo & no subhalo

        info = {
            "raw": raw,
            "halo": halo,
            "numpart_halo": numpart_halo,
            "num_sh": n_sh_in_h,
            "shids_in_hid": shids_hid,
            "numpart_shs": np_sh_in_h,
            "target_mask": tar,
            "numpart_true_bg": numpart_true_bg,
            # particles bound to central (& comparably large) subhalo(s)
            "numpart_added_bg": masked
        }

        return info
    
    def get_subhalo_info(self, shid):
        # TODO:
        pass
        
    def load_raw_halo(self, hid):
        """
        Raw == no sampling
        hid starts at 0

        """
        start, end = self.halo_offsets[hid], self.halo_offsets[hid + 1]
        halo = np.c_[self.P[start:end], self.V[start:end], self.U[start:end]]

        return halo  # (N, 3+3+1)
    
    def load_halo(self, hid):
        """
        Relative positions - does result in reflection (XY -> YZ -> XZ)
        TODO: Check for "edge"-cases
        TODO: Sampling
        hid starts at 0

        """
        halo = self.load_raw_halo(hid)
        # Relative positions & velocities
        halo_pos = np.asarray(self.sim['Groups/99/Group/GroupPos'])[hid]
        halo_vel = np.asarray(self.sim['Groups/99/Group/GroupVel'])[hid]
        halo[:, :3] = halo_pos - halo[:, :3]
        halo[:, :3] = halo_vel - halo[:, 3:6]
        
        # Potential - Neg to Pos, TODO:
        halo[:, 6] = -halo[:, 6]

        return halo  # (N, 3+3+1)
    
    def load_raw_subhalo(self, shid):
        """
        Raw == no sampling
        shid starts at 0

        """
        start, end = self.subhalo_offsets[shid], self.subhalo_offsets[shid + 1]
        subhalo = np.c_[self.P[start:end], self.V[start:end], self.U[start:end]]

        return subhalo  # (N, 3+3+1)
    
    def shids_in_hid(self, hid):
        """
        Returns shids in given hid

        """
        sh_in_h = np.asarray(self.sim['Groups/99/Subhalo/SubhaloGrNr'])
        shids_in_hid = np.where(sh_in_h == hid)[0]

        return shids_in_hid
    
    def num_sh_in_h(self, hid):
        """
        Returns number of subhalos in halo
        
        """
        num_sh_in_h = np.asarray(self.sim['Groups/99/Group/GroupNsubs'])

        return num_sh_in_h[hid]
    
    def numpart_sh_in_h(self, hid):
        """
        Returns number of particles in subhalos in given halo

        """
        sh_lens = np.asarray(self.sim['Groups/99/Subhalo/SubhaloLen'])

        return sh_lens[self.shids_in_hid(hid)]
    
    def get_mask(self, hid, anom_thresh=0.5):
        """
        Returns the mask for creating targets

        """
        np_sh_in_h = self.numpart_sh_in_h(hid)
        # fg here refers to particles not bound to the halo itself
        # This is not the FG that is used in training,
        # since the central subhalo is considered BG
        fg_end = np_sh_in_h.sum()
        ratios = np_sh_in_h[1:] / np_sh_in_h[:-1]
        to_mask = np_sh_in_h[0]
        for i, ratio in enumerate(ratios):
            if ratio >= anom_thresh:
                to_mask += np_sh_in_h[i + 1]
                continue
            break
        
        return to_mask, fg_end

    def prep_tar(self, hid, anom_thresh=0.5):
        """
        Prepares target
        If ratio of numparts >= anom_thresh (0.5 by default),
        that subhalo is included in the background.
        This is done sequentially and so,
        large similarly-sized true subhalos are not counted as background.
        
        1. / 0 => FG
        2. / 1 => BG

        """
        halo_arr = self.load_raw_halo(hid)
        tar = np.ones(halo_arr.shape[0], dtype=float)
        to_mask, fg_end = self.get_mask(hid, anom_thresh)

        # Prepping target
        tar[:to_mask] = 2.
        tar[fg_end:] = 2.

        numpart_true_bg = halo_arr.shape[0] - fg_end

        return tar, (to_mask, numpart_true_bg)

    def plot_halo(self, hid, sampled=None):
        # TODO: Sample
        # if hid is None:
        #     halo = sampled
        halo = self.load_raw_halo(hid).T # Transposing here for ease-of-use later
        tar, masked = self.prep_tar(hid)
        num_masked = sum(masked)

        fig = go.Figure(data=[
            go.Scatter3d(
                x=halo[0],
                y=halo[1],
                z=halo[2],
                mode='markers',
                marker=dict(
                    size=1, # Larger than surrounding data-points
                    color=tar,
                    opacity=0.75,
                    showscale=True,
                ))
        ])
        fig.update_layout(
            title=f"Halo ID {hid} - {halo.shape[1]} points | BG: {num_masked} points", title_x=0.5,
        )

        return fig
    
    def plot_posvel(self, hid, point_range=(0, 500)):
        """
        Plots positions (Scatter3D) and velocities (Cone)
        in `point_range` ((0, 500), by default)
        for raw and modified halo

        """
        halo = self.load_raw_halo(hid)
        halo_mod = self.load_halo(hid)
        tar, _ = self.prep_tar(hid)

        fig = make_subplots(
            rows=2, cols=2,
            specs=[[{'type': 'Scatter3d'}, {'type': 'Scatter3d'}], [{'type': 'Cone'}, {'type': 'Cone'}]])

        fig.add_trace(go.Scatter3d(
            x=halo[slice(*point_range), 0],
            y=halo[slice(*point_range), 1],
            z=halo[slice(*point_range), 2],
            mode='markers',
            marker=dict(
                size=1,
                color=tar,
                opacity=0.75,
                # showscale=True,
            )
        ), row=1, col=1)

        fig.add_trace(go.Scatter3d(
            x=halo_mod[slice(*point_range), 0],
            y=halo_mod[slice(*point_range), 1],
            z=halo_mod[slice(*point_range), 2],
            mode='markers',
            marker=dict(
                size=1,
                color=tar,
                opacity=0.75,
                # showscale=True,
            )
        ), row=1, col=2)

        fig.add_trace(go.Cone(
            x=halo[slice(*point_range), 0],
            y=halo[slice(*point_range), 1],
            z=halo[slice(*point_range), 2],
            u=halo[slice(*point_range), 3],
            v=halo[slice(*point_range), 4],
            w=halo[slice(*point_range), 5],
            # colorscale='Blues',
            showscale=False,
            sizemode="absolute",
            sizeref=100
        ), row=2, col=1)

        fig.add_trace(go.Cone(
            x=halo_mod[slice(*point_range), 0],
            y=halo_mod[slice(*point_range), 1],
            z=halo_mod[slice(*point_range), 2],
            u=halo_mod[slice(*point_range), 3],
            v=halo_mod[slice(*point_range), 4],
            w=halo_mod[slice(*point_range), 5],
            # colorscale='Blues',
            showscale=False,
            sizemode="absolute",
            sizeref=100
        ), row=2, col=2)

        fig.update_layout(
            title_text=f"Absolute and relative positions and velocities for HID: {hid} in range {point_range}.",
            title_x=0.5,
            height=1000,
        )

        # fig.update_layout(scene=dict(aspectratio=dict(x=1, y=1, z=0.8),
        #                              camera_eye=dict(x=1.2, y=1.2, z=0.6)))

        return fig


# class TNGHalo(TNGData):
#     def __init__(self, dataPath):
#         super().__init__(dataPath)
        
    