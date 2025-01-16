from bsb import LocationTargetting, config
from lfpykit import CellGeometry, LineSourcePotential
import numpy as np

from .membrane_current_recorder import MembraneCurrentRecorder


@config.node
class LFPRecorder(MembraneCurrentRecorder, classmap_entry="lfp_recorder"):
    locations = config.attr(type=LocationTargetting, default={"strategy": "everywhere"})
    x_s = np.ones(10) * 50
    y_s = np.ones(10) * 50
    z_s = np.arange(10) * 20
    sigma = 0.3

    def implement(self, adapter, simulation, simdata):
        for model, pop in self.targetting.get_targets(
            adapter, simulation, simdata
        ).items():

            origins = simdata.placement[model].load_positions()
            for target in pop:
                # collect all locations from the target cell
                locations = self.locations.get_locations(target)
                n_locs = len(locations)
                x_i = np.zeros([n_locs, 2])
                y_i = np.zeros([n_locs, 2])
                z_i = np.zeros([n_locs, 2])
                d_i = np.zeros([n_locs, 2])

                for i_loc, location in enumerate(locations):
                    # get for each location xyz coords and diam
                    section = location.section
                    idx_loc = section.locations.index(
                        location._loc
                    )  # index in section.location
                    idx_next_loc = (
                        idx_loc + 1 if location.arc(0) < 1 else idx_loc
                    )  # there is another point after
                    x_i[i_loc] = [section.x3d(idx_loc), section.x3d(idx_next_loc)]
                    y_i[i_loc] = [section.y3d(idx_loc), section.y3d(idx_next_loc)]
                    z_i[i_loc] = [section.z3d(idx_loc), section.z3d(idx_next_loc)]
                    d_i[i_loc] = [section.diam3d(idx_loc), section.diam3d(idx_next_loc)]

                # note: recording by default done section(loc.arc(0))
                # create CellGeometry of targer by using the selected locations
                # matrix M (given the probe geometry/properties)
                origin = origins[target.id]
                print('origin cell ', target.id, ' ', origin)
                cell_i = CellGeometry(
                    x=x_i + origin[0], y=y_i + origin[1], z=z_i + origin[2], d=d_i
                )
                lsp = LineSourcePotential(
                    cell_i, x=self.x_s, y=self.y_s, z=self.z_s, sigma=self.sigma
                )
                M_i = lsp.get_transformation_matrix()
                pos_nan = np.isnan(
                    np.sum(M_i, 0)
                )  # check for nan, this happens when points with 0 length

                for i_loc, location in enumerate(locations):
                    if ~pos_nan[i_loc]:
                        super()._add_imem_recorder(
                            simdata.result,
                            location,
                            name=self.name,
                            cell_type=target.cell_model.name,
                            cell_id=target.id,
                            loc=location._loc,
                            M=M_i[:, i_loc],  # .reshape([M_i.shape[0],1]),
                        )
                # pass M through the device
                # find where to apply it, nb it can be applied at each time point and del the simulate signal (keep only V_ex)
