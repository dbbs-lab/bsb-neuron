import typing

import MEAutility.core as mu
import numpy as np
from bsb import LocationTargetting, config, types
from lfpykit import CellGeometry, LineSourcePotential, RecMEAElectrode

from .membrane_current_recorder import MembraneCurrentRecorder


@config.node
class MeaElectrode:
    electrode_name = config.attr(type=str, required=True)
    definitions: dict[typing.Any] = config.dict(type=types.any_())
    rotations = config.dict(type=types.or_(types.list(type=int), float), default=None)
    shift = config.list(type=int, default=None)

    def __boot__(self):
        if self.electrode_name in mu.return_mea_list():
            self.custom = False
        else:
            if self.definitions:
                self.custom = True
                self.definitions["electrode_name"] = self.electrode_name

            else:
                raise ValueError(
                    f"Do not find {self.electrode_name} probe. Available models for MEA arrays: {mu.return_mea_list()}"
                )

    def return_probe(self):
        # Check if we are using a custom probe and create MEA object
        if self.custom:
            # Clean definitions, make sure that scaffold objects are not passed to MEA classes
            info_dict = {}
            for key, value in self.definitions.items():
                if key not in ["scaffold", "_config_parent"]:
                    info_dict[key] = value
            pos = mu.get_positions(info_dict)
            if mu.check_if_rect(info_dict):
                mea_obj = mu.RectMEA(positions=pos, info=info_dict)
            else:
                mea_obj = mu.MEA(positions=pos, info=info_dict)
        else:
            mea_obj = mu.MEA.return_mea(self.electrode_name)
        # If a rotation is selected rotate the array
        if self.rotations:
            mea_obj.rotate(self.rotations["axis"], self.rotations["angle"])
        if self.shift:
            mea_obj.move(self.shift)
        return mea_obj


@config.node
class LFPRecorder(MembraneCurrentRecorder, classmap_entry="lfp_recorder"):
    locations = config.attr(type=LocationTargetting, default={"strategy": "everywhere"})
    mea_electrode = config.attr(type=MeaElectrode, required=True)

    def implement(self, adapter, simulation, simdata):
        my_probe = self.mea_electrode.return_probe()
        for model, pop in self.targetting.get_targets(
            adapter, simulation, simdata
        ).items():

            origins = simdata.placement[model].load_positions()
            for local_cell_id, target in enumerate(pop):
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
                # create CellGeometry of target by using the selected locations
                # matrix M (given the probe geometry/properties)
                # local_cell_id = simdata.placement[model].convert_to_local(target.id)[0]
                origin = origins[local_cell_id]
                print("origin cell ", target.id, " ", origin)
                # print(
                #     f"Rank: {MPI.get_rank()} - {target.id} l {simdata.placement[model].convert_to_local(target.id)} - pop: {len(origins)}"
                # )
                cell_i = CellGeometry(
                    x=x_i + origin[0], y=y_i + origin[1], z=z_i + origin[2], d=d_i
                )
                lsp = RecMEAElectrode(
                    cell_i,
                    sigma_T=0.3,
                    sigma_S=1.5,
                    sigma_G=0.0,
                    h=400.0,
                    z_shift=-100.0,
                    method="linesource",
                    steps=20,
                    probe=my_probe,
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
                            M=M_i[:, i_loc],  # .reshape([M_i.shape[0], 1])
                        )
                # pass M through the device
                # find where to apply it, nb it can be applied at each time point and del the simulate signal (keep only V_ex)
