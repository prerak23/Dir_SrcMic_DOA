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

        # STARSS22 mic array
        dcase_arr = np.zeros((3, 2))
        dcase_arr[:, 0] = [0.02432757, 0.02432757, 0.02409021]
        dcase_arr[:, 1] = [0.02432757, -0.02432757, -0.02409021]

        panel_1_mic_1 = np.matmul(rotation_matrix, dcase_arr[:, 0])
        panel_1_mic_2 = np.matmul(rotation_matrix, dcase_arr[:, 1])

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
                        random.uniform(0.2, (room_dimension[0] - saftey_distance)), 3
                    ),
                    round(
                        random.uniform(0.2, (room_dimension[1] - saftey_distance)), 3
                    ),
                    round(
                        random.uniform(0.2, (room_dimension[2] - saftey_distance)), 3
                    ),
                ]
            )

            # Rotation of the microphone array (x-z) direction parallel to the ground.

            azimuth_rotation, elevation_rotation = self.rotation_directivity()
            # rot=random.randint(0,360)
            barycenter_ypr = [
                0,
                0,
                0,
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
            li_bc[x, :] = barycenter
            li_bc_mic[x, :] = mic_bc
            mic_pos_ypr[x, 0, :] = [0, 0]
            mic_pos_ypr[x, 1, :] = [0, 0]

        return mic_pos, li_bc, li_bc_mic, mic_pos_ypr
        # Rotation of the directivity pattern, every microphone will have a different directivity, thus there would be (3,3) rotation every micrphone array will have one directivity assoiciated with it for.

        # li_ypr=np.array([[random.randint(-180,180),random.randint(0,180)] for x in range(5)]).reshape(5,2)
