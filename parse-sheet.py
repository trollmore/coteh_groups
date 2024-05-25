from fuzzywuzzy import fuzz

WORDCOUNT_CODE = {
    "Up to 2&000": 0,
    "Between 2&000 and 4&000": 1,
    "Between 4&000 and 6&000": 2,
    "Over 6&000": 3,
    "I don't care& I just like pre-reading stuff!": 4,
}
# the max score is 9, so this allows us to normalize a scale to 10
WORDCOUNT_SCALE = 10 / (len(WORDCOUNT_CODE) - 2) ** 2

GENRE_CODE = {
    "Fantasy": 0,
    "Science Fiction": 1,
    "Horror": 2,
    "Romance": 3,
    "I have several stories, and I'd like to talk it over with my group!": 4,
    "several": 4,
}

SIZE_CODE = {"Pair": 2, "Small Group (3)": 3, "Small Group (4)": 4}


SORT_WEIGHTS = {
    "group_veto": 1000,
    "friend_req": 100,
    "wordcount": 20,
    "cw_trauma": 18,
    "cw_swearing": 15,
    "group_size": 10,
    "genre": 5,
}


class Person:
    def __init__(self, name: str, **kwargs) -> None:
        self.name = name.replace(" ", "")
        self.is_minor = kwargs["is_minor"]
        self.fb_wanted = kwargs["fb_wanted"]
        self.fb_provide = kwargs["fb_provide"]
        self.words_wr = kwargs["words_wr"]
        self.words_r = kwargs["words_r"]
        self.genre_wr = kwargs["genre_wr"]
        self.genre_r = kwargs["genre_r"]
        self.cw_wr = kwargs["cw_wr"]
        self.cw_veto = kwargs["cw_veto"] if "cw_veto" in kwargs else []
        self.size_pref = kwargs["size_pref"]
        self.team_pref = kwargs["team_pref"]
        self.team_veto = kwargs["team_veto"]
        self.seasonal = kwargs["seasonal"] if "seasonal" in kwargs else {}

    def word_dist(self, other):
        my_diff = self.words_wr - other.words_r
        their_diff = other.words_wr - self.words_r
        word_diff = (max(my_diff, 0) + max(their_diff, 0)) ** 2
        word_diff *= WORDCOUNT_SCALE

        return word_diff

    def gen_dist(self, other):
        total = 0
        total = self.word_dist(other) * SORT_WEIGHTS["wordcount"]

        return total

    def __repr__(self) -> str:
        return "\n".join(
            [
                self.name,
                f"{'minor' if self.is_minor else 'not minor'}",
                self.fb_wanted,
                str(self.fb_provide),
                str(self.words_wr),
                str(self.words_r),
                str(self.genre_wr),
                str(self.genre_r),
                str(self.cw_wr),
                str(self.cw_veto),
                str(self.size_pref),
                self.team_pref,
                self.team_veto,
            ]
        )


class Group:
    def __init__(self, members=None) -> None:
        self.members = members if members else []

    @property
    def size(self):
        return len(self.members)

    @property
    def max_size(self):
        return min([max(p.size_pref) for p in self.members] + [4])

    @property
    def max_output(self):
        return max([p.words_wr for p in self.members])

    def permute_with_member(self, person):
        new_members = self.members.copy()
        new_members.append(person)
        return Group(new_members)

    def merge_with_group(self, other):
        self.members = self.members + other.members.copy()

    def compatibility(self):
        tmp = self.members.copy()
        score = 0
        discount = ((self.size) * (self.size - 1)) / 2

        if self.size != 3:
            score += SORT_WEIGHTS["group_size"]

        while tmp:
            p: Person = tmp.pop()

            score += 0 if self.size not in p.size_pref else SORT_WEIGHTS["group_size"]

            for other in tmp:
                score += p.gen_dist(other) / discount

        return round(score, 2)

    def __repr__(self) -> str:
        return f"<<Group of size {self.size} with members {[mem.name for mem in self.members]} and fit score {self.compatibility()}>>"


class Model:
    def __init__(self, groups=None, users=None) -> None:
        self.groups = groups if groups else []
        self.users = users if users else []

    @property
    def total_fit(self) -> float:
        return sum(0, [g.compatibility() for g in self.groups])

    @property
    def avg_fit(self) -> float:
        return round(self.total_fit / len(self.groups), 2)

    def sort_users(self, func) -> None:
        self.users.sort(key=func)

    




    def __repr__(self) -> str:
        return f"total fit {self.total_fit}, avg fit {self.avg_fit}"







































def clean_data(raw_data):
    # checksum for quotes
    if raw_data.count('"') % 2 != 0:
        criminal = raw_data.split(",")[1]
        raise ValueError(
            f'Malformed input: odd number of quote characters ( " ) in response data for {criminal}.'
        )

    # replace commas within responses
    result: str = raw_data
    result = result.replace("&", " ")

    replace_mode = False
    for idx in range(len(raw_data)):
        curr_char = raw_data[idx]
        if curr_char == '"':
            replace_mode = not replace_mode
        if curr_char == "," and replace_mode:
            result = result[:idx] + "&" + result[idx + 1 :]

    # one of the question responses has a weird comma in it, which we get rid of
    result = result.replace(
        "I have several stories& and I'd like to talk it over with my group!", "several"
    )
    result = result.replace('"', "")

    return result


def translate_fb_pref(fb_str):
    return (
        "Writer" if "Writer" in fb_str else "Reader" if "Reader" in fb_str else "Hype"
    )


#######################################################
#                   begin script                      #
#######################################################

users = []

with open("source.csv", "r", encoding="utf-8") as file:
    file.readline()  # clear header row
    for response in file:

        cleaned = clean_data(response).split(",")

        name = cleaned[1]

        # age
        is_minor = "Under" in cleaned[2]

        # what kind of feedback are you looking for?
        fb_wanted = translate_fb_pref(cleaned[3])

        # how much output will you bring?
        words_wr = WORDCOUNT_CODE[cleaned[4]]

        # what genre do you write?
        genre_wr = cleaned[5].split("& ")

        # what content warnings apply to your story?
        cw_wr = [
            GENRE_CODE[cw] if cw in GENRE_CODE else cw for cw in cleaned[6].split("& ")
        ]

        # what size group are you okay with?
        size_pref = [SIZE_CODE[pref] for pref in cleaned[7].split("& ")]

        # what kind of feedback are you okay giving?
        fb_provide = [translate_fb_pref(cleaned[8])]

        # how many words can you commit to critiquing each week?
        words_r = WORDCOUNT_CODE[cleaned[9]]

        # what genres will you be bringing to group?
        genre_r = [
            GENRE_CODE[genre] if genre in GENRE_CODE else genre
            for genre in cleaned[10].split("& ")
        ]

        # which content warnings do you want to avoid?
        cw_veto = (
            []
            if cleaned[11] == "I'll read anything!"
            else [
                GENRE_CODE[cw] if cw in GENRE_CODE else cw
                for cw in cleaned[11].split("& ")
            ]
        )

        # who would you like to be with?
        team_pref = cleaned[12]

        # who do you want to avoid?
        team_veto = cleaned[13]

        user = Person(
            name,
            is_minor=is_minor,
            fb_wanted=fb_wanted,
            words_wr=words_wr,
            genre_wr=genre_wr,
            cw_wr=cw_wr,
            size_pref=size_pref,
            fb_provide=fb_provide,
            words_r=words_r,
            genre_r=genre_r,
            cw_veto=cw_veto,
            team_pref=team_pref,
            team_veto=team_veto,
        )

        users.append(user)

users.sort(key=lambda p: max(p.size_pref))
users.sort(key=lambda p: p.words_r)
users.sort(key=lambda p: p.words_wr)

# print([max(user.size_pref) for user in users])

groups = []
curr_group: Group = Group([])
# users = users[:4]
while users:
    p = users.pop(0)

    if curr_group.size < max(p.size_pref) and curr_group.size < curr_group.max_size:
        curr_group = curr_group.permute_with_member(p)
    else:
        groups.append(curr_group)
        curr_group = Group([p])
    # print(curr_group.compatibility())

groups.append(curr_group)


# print(groups[0].compatibility())

for group in groups:
    print(group)


# print(users[4].words_r)
# print(users[4].words_wr)
# print(users[5].words_r)
# print(users[5].words_wr)
# print(users[4].word_dist(users[5]))
