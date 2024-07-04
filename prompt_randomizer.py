import random


def prompt_randomize ():

    gender_list = ["man", "woman" , "non-binary person", "transgender person"]
    job_list = ["businessman", "doctor", "student", "lawyer", "mechanic", "engineer", "librarian", "manager", "designer", "worker", "artist", "researcher", "banker", "economist", "farmer", "carpernter", "chemist"]
    skincolor_list = ["white", "dark", "black", "caucasian", "asian", "latinx", "indigenous", "multiracial", "hispanic", "pacific", "south-east asian", "north-african", "south-african", "south-american"]

    height_list = ["tall", "short", "average"]

    hair_list = ["long", "short", "curly", "very short", "very long", "pony tail", "bald"]

    hair_color_list = ["brown", "black", "red", "blond", "white"]

    color_cloths_list = ["black", "white", "red", "blue", "green", "yellow", "orange", "purple", "pink"]

    bottom_cloths_list = ["trousers", "skirt", "trackpants", "dress", "overalls", "jens", "suit"]

    top_cloths_list = ["shirt", "jacket", "blouse", "sweater", "tank-top", "cardigan"]

    shoes_list = ["shoes", "boots", "snickers", "heels"]

    activity_list =["walking", "running", "jogging"]

    body_shape_list = ["midweight", "overweight", "stout", "buff", "burly", "fit", "flyweight", "skinny", "not skinny", "slender", "slim"]



    visual_prompt = f'"<ali-1-4-CCTV>", "(small hips)++", "(fully clothed)++", "{random.choice(skincolor_list)}", "{random.choice(gender_list)}", "{random.choice(height_list)} height", "{random.choice(body_shape_list)}", "({random.choice(job_list)})++", "{random.choice(hair_list)} hair", "{random.choice(color_cloths_list)} {random.choice(bottom_cloths_list)}", "{random.choice(color_cloths_list)} {random.choice(top_cloths_list)}", "{random.choice(shoes_list)}", "{random.choice(activity_list)}"'

    positive_prompt = f'[{visual_prompt} ,"(Not that colorful)++", "(muted colors)++", "(desaturated)+", "(pure white background)+","(blank background)+++", "clean background","highly detailed","4k photo", "(not sporty)", "(fully clothed)"].and()'

    return positive_prompt
