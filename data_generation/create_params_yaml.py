import yaml

dict = {
    "Number_of_rooms": 40000,
    "Realistic_walls": True,
    "fs": 16000,
    "reference_freq": 125,
    "air_absorption": True,
    "max_order": 20,
    "ray_tracing": False,
    "min_phase": True,
    "humidity": 0.42,
    "temprature": 20.0,
    "no_of_rec_room": 3,
    "no_of_src_room": 1,
    "saftey_distance": 0.2,
    "source_dir_list": [],
    "receiver_dirs": [],
    "name_of_the_dataset": "0000",
}

with open("params.yml", "w") as f:
    yaml.dump(dict, f)
