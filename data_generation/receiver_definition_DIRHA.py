import numpy as np
import random
from generate_simulator_params import cart2sphere

# Generate receivers position
class util_receiver:
    def __init__(self):
        self.dic_receiver = {}

    def rotation_directivity(self):
        m_x = np.random.normal(loc=0, scale=1, size=1)
        m_y = np.random.normal(loc=0, scale=1, size=1)
        m_z = np.random.normal(loc=0, scale=1, size=1)

        abs_ = np.sqrt(m_x ** 2 + m_y ** 2 + m_z ** 2)

        # cart to deg
        azi_, col_ = cart2sphere((m_x / abs_), (m_y / abs_), (m_z / abs_))

        return azi_, col_

    def mic_defination_array(self, rotation_matrix, barycenter):

        # Voicehome mic array
        voicehome2 = np.zeros((3, 2))
        voicehome2[:, 0] = [0.15, 0, 0]
        voicehome2[:, 1] = [-0.15, 0, 0]
        panel_1_mic_1 = np.matmul(rotation_matrix, voicehome2[:, 0])
        panel_1_mic_2 = np.matmul(rotation_matrix, voicehome2[:, 1])

        return (
            panel_1_mic_1 + barycenter,
            panel_1_mic_2 + barycenter,
            ((panel_1_mic_1 + barycenter) + (panel_1_mic_2 + barycenter)) / 2,
        )

    def generate_receivers_rooms(
        self,
        room_dimension,
        different_no_receivers,
        saftey_distance,
        room_id,
        mic_in_ula=None,
    ):
        a = different_no_receivers

        li_bc = np.empty((a, 3))  # list for the bary center
        # li_ypr=np.empty((a,2)) # list for the ypr of the barycenter
        mic_pos = np.empty((a, 2, 3))  # viewpoints x mics x coordinate
        li_bc_mic = np.empty((a, 3))
        mic_pos_ypr = np.empty((a, 2, 2))  # viewpoints x mics x directivity ypr

        for x in range(different_no_receivers):

            # Random barycenter coordinates selection , within the saftey distance from the walls of the room

            barycenter = np.array(
                [
                    round(
                        random.uniform(0.4, (room_dimension[0] - saftey_distance)), 3
                    ),
                    round(
                        random.uniform(0.4, (room_dimension[1] - saftey_distance)), 3
                    ),
                    round(
                        random.uniform(0.4, (room_dimension[2] - saftey_distance)), 3
                    ),
                ]
            )

            pick_random_surface = np.random.randint(3)
            if pick_random_surface == 0:
                seq = [1, 2]
            elif pick_random_surface == 1:
                seq = [0, 2]
            else:
                seq = [0, 1]

            wall_switch = np.random.randint(2)
            barycenter_copy = np.copy(barycenter)

            if wall_switch == 0:
                barycenter_copy[pick_random_surface] = 0
            else:
                barycenter_copy[pick_random_surface] = (
                    room_dimension[pick_random_surface] - 0.05
                )

            mic_pos[x, 0, :] = barycenter_copy

            seq_no = np.random.randint(2)
            seq_id = seq[seq_no]

            if barycenter_copy[seq_id] + 0.30 < room_dimension[seq_id]:
                barycenter_copy[seq_id] += 0.30
            else:
                if seq_no == 0:
                    seq_id = seq[1]
                    barycenter_copy[seq_id] += 0.30
                else:
                    seq_id = seq[0]
                    barycenter_copy[seq_id] += 0.30

            mic_pos[x, 1, :] = barycenter_copy

            mic_bc = (mic_pos[x, 0, :] + mic_pos[x, 1, :]) / 2
            # Rotation of the microphone array (x-z) direction parallel to the ground.

            """
            azimuth_rotation_bc, elevation_rotation_bc = self.rotation_directivity()


            barycenter_ypr = [
                random.randint(0, 360),
                0,
                elevation_rotation_bc,
            ]  # random.randint(0,360) Can also be selected from the unit vector selection.

            y, p, r = barycenter_ypr[0], barycenter_ypr[1], barycenter_ypr[2]
            rotation_mat_1 = np.array(
                [
                    np.cos(np.pi * (y) / 180) * np.cos(np.pi * (p) / 180),
                    np.cos(np.pi * (y) / 180)
                    * np.sin(np.pi * (p) / 180)
                    * np.sin(np.pi * (r) / 180)
                    - np.sin(np.pi * (y) / 180) * np.cos(np.pi * (r) / 180),
                    np.cos(np.pi * (y) / 180)
                    * np.sin(np.pi * (p) / 180)
                    * np.cos(np.pi * (r) / 180)
                    + np.sin(np.pi * (y) / 180) * np.sin(np.pi * (r) / 180),
                ]
            )
            rotation_mat_2 = np.array(
                [
                    np.sin(np.pi * (y) / 180) * np.cos(np.pi * (p) / 180),
                    np.sin(np.pi * (y) / 180)
                    * np.sin(np.pi * (p) / 180)
                    * np.sin(np.pi * (r) / 180)
                    + np.cos(np.pi * (y) / 180) * np.cos(np.pi * (r) / 180),
                    np.sin(np.pi * (y) / 180)
                    * np.sin(np.pi * (p) / 180)
                    * np.cos(np.pi * (r) / 180)
                    - np.cos(np.pi * (y) / 180) * np.sin(np.pi * (r) / 180),
                ]
            )
            rotation_mat_3 = np.array(
                [
                    -np.sin(np.pi * (p) / 180),
                    np.cos(np.pi * (p) / 180) * np.sin(np.pi * (r) / 180),
                    np.cos(np.pi * (p) / 180) * np.cos(np.pi * (r) / 180),
                ]
            )
            rotation_mat = np.array(
                [rotation_mat_1, rotation_mat_2, rotation_mat_3]
            )  # 3*3 rotation matrice.

            mic_pos_1, mic_pos_2, mic_bc = self.mic_defination_array(
                rotation_mat, barycenter
            )
            mic_pos[x, 0, :] = mic_pos_1
            mic_pos[x, 1, :] = mic_pos_2
            """

            li_bc[x, :] = barycenter
            li_bc_mic[x, :] = mic_bc
            mic_pos_ypr[x, 0, :] = self.rotation_directivity()
            mic_pos_ypr[x, 1, :] = self.rotation_directivity()

        return mic_pos, li_bc, li_bc_mic, mic_pos_ypr
        # Rotation of the directivity pattern, every microphone will have a different directivity, thus there would be (3,3) rotation every micrphone array will have one directivity assoiciated with it for.

        # li_ypr=np.array([[random.randint(-180,180),random.randint(0,180)] for x in range(5)]).reshape(5,2)
