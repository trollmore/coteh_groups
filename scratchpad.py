from fuzzywuzzy import fuzz


# print(fuzz.partial_ratio("BadgerPunch with @loop0s and @mawiscool but we haven't had much activity and could maybe do with new blood to reinvigorate the group (we discussed this)","Trollmore"))
# print(fuzz.partial_ratio(,"SaithorthePyro"))
# print(fuzz.partial_ratio("BadgerPunch with @loop0s and @mawiscool but we haven't had much activity and could maybe do with new blood to reinvigorate the group (we discussed this)","Queen2880andpielord"))

NAMES = [
    "Trollmore",
    "Queen2880andpielord",
    "SaithorthePyro",
    "Astelian",
    "Mawiscool"
    ]

FRIENDS = [
    "",
    "BadgerPunch with @loop0s and @mawiscool but we haven't had much activity and could maybe do with new blood to reinvigorate the group (we discussed this)",
    "Saithorthepyro",
    "Whichever, you know I'm good for it",
    "@saithorthepyro @dionsky"
]

for name in NAMES:
    for friend in FRIENDS:
        ratio = fuzz.partial_ratio(name, friend)
        if ratio > 80:
            print(name, friend)
        # ratio2 = fuzz.partial_ratio(friend, name)
        # if ratio2 > 80:
        #     print(name, friend)


        hash()