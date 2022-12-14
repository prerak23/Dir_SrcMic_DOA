import numpy as np
import yaml
import random
import h5py
from scipy.spatial import distance
from tqdm import tqdm

# import receiver_defination_voicehome2
import receiver_defination_DIRHA

# import receiver_defination_starss22
from yaml.loader import SafeLoader

# Generate room configuration.
# These room configurations is used by pysofamyroom to generate RIR's.


def cart2sphere(x, y, z):
    # Convert cartesian coordinates into spherical coordinates (radians)

    r = np.sqrt(np.square(x) + np.square(y) + np.square(z))

    theta_fibo = np.arccos((z / r))

    phi_fibo = np.arctan2(y, x)

    phi_fibo = np.degrees(phi_fibo)  # phi_fibo was in range of [-np.pi,np.pi]

    phi_fibo = (phi_fibo + 360) % 360

    return phi_fibo, np.degrees(theta_fibo)


def calculate_source_doa_1d(
    rec_pos, rec_bc, src_pos, rec_bc_mic, no_of_rec_room, no_of_source_room
):  # One mic array and multiple source positions in the room.
    azimuth = []

    for x in range(no_of_source_room):
        u_vec = rec_bc_mic[0, :] - rec_pos[0, 0, :]
        v_vec = rec_bc_mic[0, :] - src_pos[x, :]

        azimuth_src = np.rad2deg(
            np.arccos(
                np.dot(u_vec, v_vec)
                / (np.sqrt(np.sum(u_vec ** 2)) * np.sqrt(np.sum(v_vec ** 2)))
            )
        )

        azimuth.append(azimuth_src)

    return np.array(azimuth)


class util_room:
    def height_width_length(self):
        # Room Dimension Range: Length (3,10), Width(3,10), Height(2.5,4)
        values = [
            round(random.uniform(2, 10), 1),
            round(random.uniform(2, 10), 1),
            round(random.uniform(2, 4.5), 1),
        ]
        surface_area = 2 * (
            values[0] * values[1] + values[1] * values[2] + values[0] * values[2]
        )

        return values, surface_area

    def get_diffusion_coeff(self):
        # Diffusion Coeff Range: [0.2,1]
        coeff = round(random.uniform(0.2, 1), 2)
        return [coeff for x in range(36)]

    def get_absorption_coeff(self, realistic_walls=False):
        # Coin-flip to get the different absorption profiles

        # Unrealistic walls D-1_0000

        if realistic_walls == False:
            abs_coeff_val = round(random.uniform(0.02, 0.50), 2)
            abs_coeff_wall = np.ones((6, 6)) * abs_coeff_val

        # Realistic walls
        # id reflective walls = 7
        # absorbant wall = 8
        else:
            no_reflective_walls = random.choice([0, 1, 2, 3, 4, 5, 6])
            walls_profile = np.array([8 for i in range(6)])
            id_reflective = np.random.choice(
                [0, 1, 2, 3, 4, 5], size=no_reflective_walls, replace=False
            )
            walls_profile[id_reflective] = 7

            abs_coeff_wall = np.empty((6, 6))
            for i, a in enumerate(walls_profile):
                if a == 7:  # Reflective Profile
                    abs_coeff_val = round(random.uniform(0.01, 0.12), 2)
                    abs_coeff_wall[i, :] = [abs_coeff_val] * 6
                elif a == 8:
                    f_o_c = random.choice([1, 2])  # Removed 0 wall profile.
                    if f_o_c == 0:
                        abs_coeff_val = [
                            round(random.uniform(0.01, 0.50), 2),
                            round(random.uniform(0.01, 0.50), 2),
                            round(random.uniform(0.01, 0.30), 2),
                            round(random.uniform(0.01, 0.12), 2),
                            round(random.uniform(0.01, 0.12), 2),
                            round(random.uniform(0.01, 0.12), 2),
                        ]
                        abs_coeff_wall[i, :] = abs_coeff_val
                    elif f_o_c == 1:
                        abs_coeff_val = [
                            round(random.uniform(0.01, 0.70), 2),
                            round(random.uniform(0.15, 1.00), 2),
                            round(random.uniform(0.40, 1.00), 2),
                            round(random.uniform(0.40, 1.00), 2),
                            round(random.uniform(0.40, 1.00), 2),
                            round(random.uniform(0.30, 1.00), 2),
                        ]
                        abs_coeff_wall[i, :] = abs_coeff_val
                    else:
                        abs_coeff_val = [
                            round(random.uniform(0.01, 0.20), 2),
                            round(random.uniform(0.01, 0.30), 2),
                            round(random.uniform(0.05, 0.50), 2),
                            round(random.uniform(0.15, 0.60), 2),
                            round(random.uniform(0.25, 0.75), 2),
                            round(random.uniform(0.30, 0.80), 2),
                        ]
                        abs_coeff_wall[i, :] = abs_coeff_val

        return abs_coeff_wall.tolist()


# Generate real and fake sources for the simulation.


class util_source:
    def generate_source_room(
        self, room_dimension, different_no_sources, saftey_distance, barycenter
    ):
        # Random source in the room
        ll_source = np.empty(
            (different_no_sources, 3)
        )  # 5 viewpoints per room so 5*3 matrice
        ypr_source = np.empty((different_no_sources, 2))

        description_source = []

        x_safteydistance = room_dimension[0] - saftey_distance

        y_safteydistance = room_dimension[1] - saftey_distance

        z_safteydistance = room_dimension[2] - saftey_distance

        for i in range(different_no_sources):
            tmp_cord = np.array(
                [
                    round(random.uniform(0.2, x_safteydistance), 3),
                    round(random.uniform(0.2, y_safteydistance), 3),
                    round(random.uniform(0.2, z_safteydistance), 3),
                ]
            )

            # 3 view point and only one fixed source for it , thus the distance for this source should be greater than 0.6 for all the receivers in the room.
            # 3 view point and with fixed micropohne and 3 different source positions.

            tmp_distance_1 = distance.euclidean(tmp_cord, barycenter[0, :])
            # tmp_distance_2=distance.euclidean(tmp_cord,barycenter[1,:])
            # tmp_distance_3=distance.euclidean(tmp_cord,barycenter[2,:])

            ###############################################################################################################################################
            # Until the distance between the source and the "Array" barycenter is greater than 0.3, we keep on finding the new coordinates for the source.#
            ###############################################################################################################################################

            while (
                tmp_distance_1 < 0.6
            ):  # and tmp_distance_2 < 0.8 and tmp_distance_3 < 0.8) :
                tmp_cord = np.array(
                    [
                        round(random.uniform(0.2, x_safteydistance), 3),
                        round(random.uniform(0.2, y_safteydistance), 3),
                        round(random.uniform(0.2, z_safteydistance), 3),
                    ]
                )
                tmp_distance_1 = distance.euclidean(tmp_cord, barycenter[0, :])
                # tmp_distance_2=distance.euclidean(tmp_cord,barycenter[1,:])
                # tmp_distance_3=distance.euclidean(tmp_cord,barycenter[2,:])

            ll_source[i, :] = tmp_cord
            # The roatation of the directivity pattern would be parrallel to the ground hence just the random roataion around azimuth.

            ypr_source[i, :] = [random.randint(-180, 180), 0]
            # description_source.append(random.sample(self.description,1))

        # Retruns source positions and the ypr roation of the source
        return ll_source, ypr_source

    def fake_source_rooom(
        self, room_dimension, different_no_sources, saftey_distance, barycenter
    ):
        # Generate fake source for same reciver positions, basically will be used to calculate the late reverberant RIR's which will be help us in diffuse noise model.

        ll_source = np.empty((different_no_sources, 3))

        # ypr_source = np.empty((different_no_sources, 2))

        description_source = []

        x_safteydistance = room_dimension[0] - saftey_distance

        y_safteydistance = room_dimension[1] - saftey_distance

        z_safteydistance = room_dimension[2] - saftey_distance

        for i in range(different_no_sources):
            tmp_cord = np.array(
                [
                    round(random.uniform(0.3, x_safteydistance), 3),
                    round(random.uniform(0.3, y_safteydistance), 3),
                    round(random.uniform(0.3, z_safteydistance), 3),
                ]
            )
            tmp_distance_1 = distance.euclidean(tmp_cord, barycenter[0, :])
            # tmp_distance_2=distance.euclidean(tmp_cord,barycenter[1,:])
            # tmp_distance_3=distance.euclidean(tmp_cord,barycenter[2,:])

            while (
                tmp_distance_1 < 0.6
            ):  # and tmp_distance_2 < 0.6 and tmp_distance_3 < 0.6) :
                tmp_cord = np.array(
                    [
                        round(random.uniform(0.3, x_safteydistance), 3),
                        round(random.uniform(0.3, y_safteydistance), 3),
                        round(random.uniform(0.3, z_safteydistance), 3),
                    ]
                )

                tmp_distance_1 = distance.euclidean(tmp_cord, barycenter[0, :])
                # tmp_distance_2=distance.euclidean(tmp_cord,barycenter[1,:])
                # tmp_distance_3=distance.euclidean(tmp_cord,barycenter[2,:])

            ll_source[i, :] = tmp_cord
            # ypr_source[i,:] = [random.randint(-180, 180), random.randint(0,180)]
            # description_source.append(random.sample(self.description,1))

        return ll_source


class conf_files:
    def __init__(
        self,
        number_of_rooms,
        realistic_walls,
        fs,
        reference_frequency,
        air_absorption,
        max_order,
        ray_tracing,
        min_phase,
        humidity,
        temprature,
        no_of_rec_room,
        no_of_src_room,
        saftey_distance,
        name_of_the_dataset,
        source_dir_list,
        rec_dirs,
    ):
        self.number_rooms = number_of_rooms
        self.realistic_walls = realistic_walls
        self.fs = fs
        self.reference_frequency = reference_frequency
        self.air_absorption = air_absorption
        self.max_order = max_order
        self.ray_tracing = ray_tracing
        self.min_phase = min_phase
        self.humidity = humidity
        self.temprature = temprature
        self.no_of_rec_room = no_of_rec_room
        self.no_of_src_room = no_of_src_room
        self.saftey_distance = saftey_distance
        self.name_of_the_dataset = name_of_the_dataset
        self.source_dir_list = source_dir_list
        self.rec_dirs = rec_dirs

        self.receiver_file = receiver_defination_DIRHA.util_receiver()
        self.source_file = util_source()
        self.util_room = util_room()
        self.params_file()
        self.room_file()

    def params_file(self):
        dict_file = {
            "simulation_params": {
                "fs": self.fs,
                "referencefrequency": self.reference_frequency,
                "air_absorption": self.air_absorption,
                "max_order": self.max_order,
                "ray_tracing": self.ray_tracing,
                "min_phase": self.min_phase,
            }
        }
        with open("conf_sim_params.yml", "w") as file:
            documents = yaml.dump(dict_file, file)

    def room_file(self):
        dict_file = {}
        dict_file_receiver = {}
        dict_file_source = {}
        dict_file_noise_source = {}
        reference_freq = self.reference_frequency
        no_of_reciver_per_room = self.no_of_rec_room
        no_of_source_per_room = self.no_of_src_room
        saftey_distance = (
            self.saftey_distance  # Saftey distance between source and microphone with the room walls.
        )

        # rooms_already_done=30000
        for x in tqdm(range(self.number_rooms)):

            # x+=rooms_already_done

            return_dimension, sa = self.util_room.height_width_length()
            dict_file["room_" + str(x)] = {"surface": {}}
            dict_file["room_" + str(x)]["volume"] = (
                return_dimension[0] * return_dimension[1] * return_dimension[2]
            )
            dict_file["room_" + str(x)]["surface_area"] = sa

            dict_file["room_" + str(x)]["dimension"] = return_dimension
            # dict_file['room_' + str(x)]['humidity'] = humidity
            # dict_file['room_' + str(x)]['temperature'] = temprature
            dict_file["room_" + str(x)]["surface"]["center_frequency"] = [
                reference_freq * pow(2, a) for a in range(6)
            ]
            dict_file["room_" + str(x)]["surface"][
                "absorption"
            ] = self.util_room.get_absorption_coeff(self.realistic_walls)
            dict_file["room_" + str(x)]["surface"][
                "diffusion"
            ] = self.util_room.get_diffusion_coeff()

            # Generate receiver postions , array barycenter, roation of individual mics,
            (
                rec_pos,
                li_bc,
                li_bc_mic,
                rec_pos_ypr,
            ) = self.receiver_file.generate_receivers_rooms(
                return_dimension,
                no_of_reciver_per_room,
                saftey_distance,
                "room_" + str(x),
            )
            # random.choice([2,3]) for x in range(no_of_reciver_per_room)
            # li_rec_1,li_rec_2,li_bc=self.receiver_file.generate_receivers_rooms(return_dimension,no_of_reciver_per_room,saftey_distance,'room_' + str(x))

            # Generate source position and barycenter
            li_sc, li_sc_ypr = self.source_file.generate_source_room(
                return_dimension, no_of_source_per_room, saftey_distance, li_bc
            )

            # li_sc=self.source_file.generate_source_room(return_dimension,no_of_source_per_room,saftey_distance,li_bc)

            # Generate noise source

            li_nsc = self.source_file.fake_source_rooom(
                return_dimension, 1, saftey_distance, li_bc
            )

            dict_file_receiver["room_" + str(x)] = {
                "barycenter": li_bc.tolist(),
                "rec_pos": rec_pos.tolist(),
                "barycenter_mic": li_bc_mic.tolist(),
                "rec_pos_ypr": rec_pos_ypr.tolist(),
                "directivity": list(self.rec_dirs),
            }

            # dict_file_receiver['room_' + str(x)] = {'barycenter':li_bc.tolist(),'mic_pos_1':li_rec_1.tolist(),'mic_pos_2':li_rec_2.tolist()}

            # dict_file_source['room_' + str(x)]={'source_pos':li_sc.tolist(),'source_ypr':li_sc_ypr.tolist(),'directivity':random.choice(["CARDIOID","HYPERCARDIOID","SUBCARDIOID"])}

            # Calculate GT DoA
            azimuth = calculate_source_doa_1d(
                rec_pos, li_bc, li_sc, li_bc_mic, no_of_rec_room=1, no_of_source_room=3
            )

            dict_file_source["room_" + str(x)] = {
                "source_pos": li_sc.tolist(),
                "source_ypr": li_sc_ypr.tolist(),
                "theta_1d": azimuth.tolist(),
                "directivity": random.sample(
                    self.source_dir_list, k=len(self.source_dir_list)
                ),
            }

            # dict_file_source['room_' + str(x)]={'source_pos':li_sc.tolist()}

            dict_file_noise_source["room_" + str(x)] = {"source_pos": li_nsc.tolist()}

        with open("conf_room_setup_" + self.name_of_the_dataset + ".yml", "w") as file:
            documents = yaml.dump(dict_file, file)
        with open("conf_receivers_" + self.name_of_the_dataset + ".yml", "w") as file_1:
            documents = yaml.dump(dict_file_receiver, file_1)
        with open("conf_source_" + self.name_of_the_dataset + ".yml", "w") as file_2:
            documents = yaml.dump(dict_file_source, file_2)
        with open(
            "conf_noise_source_" + self.name_of_the_dataset + ".yml", "w"
        ) as file_3:
            documents = yaml.dump(dict_file_noise_source, file_3)


if __name__ == "__main__":

    def generate_parameters():

        with open("params.yml", "r") as f:
            data = yaml.load(f, Loader=SafeLoader)
            number_of_rooms = data["Number_of_rooms"]
            realistic_walls = data["Realistic_walls"]
            fs = data["fs"]
            reference_frequency = data["reference_freq"]
            air_absorption = data["air_absorption"]
            max_order = data["max_order"]
            ray_tracing = data["ray_tracing"]
            min_phase = data["min_phase"]
            humidity = data["humidity"]
            temprature = data["temprature"]
            no_of_rec_room = data["no_of_rec_room"]
            no_of_src_room = data["no_of_src_room"]
            saftey_distance = data["saftey_distance"]
            name_of_the_dataset = data["name_of_the_dataset"]
            source_dir_list = data["source_dir_list"]
            rec_dirs = data["receiver_dirs"]

        conf_files(
            number_of_rooms,
            realistic_walls,
            fs,
            reference_frequency,
            air_absorption,
            max_order,
            ray_tracing,
            min_phase,
            humidity,
            temprature,
            no_of_rec_room,
            no_of_src_room,
            saftey_distance,
            name_of_the_dataset,
            source_dir_list,
            rec_dirs,
        )

    generate_parameters()


# Extra Chunk of code comments
"""
def cart2sphere(points):
    # Convert cartesian coordinates into spherical coordinates (radians)

    r = np.sqrt(np.square(points[:, 0]) + np.square(points[:, 1]) + np.square(points[:, 2]))

    theta_fibo = np.arccos((points[:, 2] / r))

    phi_fibo = np.arctan2(points[:, 1], points[:, 0])


    #phi_fibo += np.pi  # phi_fibo was in range of [-np.pi,np.pi]

    return phi_fibo,theta_fibo,r





More microphone on the array mic_defination_array
mic_3=np.matmul(rotation_matrix,np.array([(distance_wrt_bc+distance_wrt_bc),0,0]))
mic_4=np.matmul(rotation_matrix,np.array([-(distance_wrt_bc+distance_wrt_bc),0,0]))
mic_5 = np.matmul(rotation_matrix, np.array([(distance_wrt_bc)*3, 0, 0]))
mic_6 = np.matmul(rotation_matrix, np.array([-(distance_wrt_bc)*3, 0, 0]))
mic_7 = np.matmul(rotation_matrix, np.array([(distance_wrt_bc)*4, 0, 0]))
mic_8 = np.matmul(rotation_matrix, np.array([-(distance_wrt_bc)*4, 0, 0]))
mic_9 = np.matmul(rotation_matrix, np.array([(distance_wrt_bc)*5, 0, 0]))
mic_10 = np.matmul(rotation_matrix, np.array([-(distance_wrt_bc)*5, 0, 0]))
mic_3+barycenter,mic_4+barycenter,mic_5+barycenter,mic_6+barycenter,mic_7+barycenter,mic_8+barycenter,mic_9+barycenter,mic_10+barycenter

In generate receiver_room

mic_pos_3_arr = np.empty((a,3))
mic_pos_4_arr = np.empty((a,3))
mic_pos_5_arr = np.empty((a, 3))
mic_pos_6_arr = np.empty((a, 3))
mic_pos_7_arr = np.empty((a, 3))
mic_pos_8_arr = np.empty((a, 3))
mic_pos_9_arr = np.empty((a, 3))
mic_pos_10_arr = np.empty((a, 3))

mic_pos_3,mic_pos_4,mic_pos_5,mic_pos_6,mic_pos_7,mic_pos_8,mic_pos_9,mic_pos_10
mic_pos_3_arr[x,:]=mic_pos_3
mic_pos_4_arr[x,:]=mic_pos_4
mic_pos_5_arr[x, :] = mic_pos_5
mic_pos_6_arr[x, :] = mic_pos_6
mic_pos_7_arr[x, :] = mic_pos_7
mic_pos_8_arr[x, :] = mic_pos_8
mic_pos_9_arr[x, :] = mic_pos_9
mic_pos_10_arr[x, :] = mic_pos_10
mic_pos_3_arr,mic_pos_4_arr,mic_pos_5_arr,mic_pos_6_arr,mic_pos_7_arr,mic_pos_8_arr,mic_pos_9_arr,mic_pos_10_arr

#self.dic_receiver[room_id]={'barycenter':[barycenter,barycenter_ypr],'ll_receiver':li_receiver}
#self.dic_receiver[room_id]['ll_receiver']=[mic_pos_1,mic_pos_2]
#dic_receiver[room_id]['orientation']=[[0,0,0],[0,0,0]]
#dic_receiver[room_id]['description']='omnidirectional'

HDF5 file
#self.file=h5py.File(name_of_the_dataset+".hdf5",'w')
 self.file.attrs['fs'] = 44100
        self.file.attrs['response_duration'] = 1.25
        self.file.attrs['bandsperoctave'] = 1
        self.file.attrs['referencefrequency'] = 125
        self.file.attrs['airabsorption'] = True
        self.file.attrs['distanceattenuation'] = True
        self.file.attrs['subsampleaccuracy'] = False
        self.file.attrs['highpasscutoff'] = 0
        self.file.attrs['verbose'] = True
        self.file.attrs['simulatespecular'] = True
        self.file.attrs['reflectionorder'] = [10,10,10]
        self.file.attrs['simulatediffuse'] = True
        self.file.attrs['numberofrays'] = 2000
        self.file.attrs['diffusetimestep'] = 0.01
        self.file.attrs['rayenergyfloordB'] = -80.0
        self.file.attrs['uncorrelatednoise'] = True

 #room=self.file.create_group("room_config")
#receiver=self.file.create_group("receiver_config")
#source=self.file.create_group("source_config")
#room.attrs['saftey_distance'] = 0.3
#room.attrs['reference_freq'] = 125
#room.attrs['no_of_receiver_per_room'] = 5

room_id=room.create_group('room_' + str(x))
receiver_id=receiver.create_group('room_'+str(x))
source_id=source.create_group('room_'+str(x))

room_id.create_dataset('dimension',3,data=return_dimension)
room_id.create_dataset('humidity',1,data=humidity)
room_id.create_dataset('temprature',1,data=temprature)
room_id.create_dataset('frequency',6,data=[reference_freq * pow(2, a) for a in range(6)])
room_id.create_dataset('absorption',36,data=self.util_room.get_absorption_coeff())
room_id.create_dataset('diffusion',36,data=self.util_room.get_diffusion_coeff())

li_rec_3,li_rec_4,li_rec_5,li_rec_6,li_rec_7,li_rec_8,li_rec_9,li_rec_10,
            ,'mic_pos_3':li_rec_3.tolist(),
'mic_pos_4':li_rec_4.tolist(),'mic_pos_5':li_rec_5.tolist(),'mic_pos_6':li_rec_6.tolist(),'mic_pos_7':li_rec_7.tolist(), ,'mic_pos_8':li_rec_8.tolist(),'mic_pos_9':li_rec_9.tolist(),'mic_pos_10':li_rec_10.tolist()


receiver_id.create_dataset('barycenter',(a,3),data=li_bc)
receiver_id.create_dataset('barycenter_ypr', (a, 3), data=li_ypr)
receiver_id.create_dataset('mic_pos_1',(a,3),data=li_rec_1)
receiver_id.create_dataset('mic_pos_2',(a,3),data=li_rec_2)
receiver_id.create_dataset('mic_pos_3', (a, 3), data=li_rec_3)
receiver_id.create_dataset('mic_pos_4', (a, 3), data=li_rec_4)
source_id.create_dataset('source_pos',(a,3),data=li_sc)
source_id.create_dataset('source_ypr', (a, 3), data=li_sc_ypr)
source_id.create_dataset('source_description', (a, 1), data=li_sc_ds)
#dict_file['room_'+str(x)]=room_dict['room_'+str(x)]


"""
