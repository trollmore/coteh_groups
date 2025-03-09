from fuzzywuzzy import fuzz
from fuzzywuzzy import process
from functools import reduce
import csv

WORDCOUNT_CODE = {
    "Up to 2,000": 0,
    "Between 2,000 and 4,000": 1,
    "Between 4,000 and 6,000": 2,
    "Over 6,000": 3,
    "I don't care, I just like pre-reading stuff!": 4,
    "": 5
}
# the max diff^2 is 9, so this allows us to normalize to a 0-10 scale
WORDCOUNT_SCALE = 10 / (len(WORDCOUNT_CODE) - 2) ** 2

GENRE_CODE = {
    "Epic Fantasy": 0,
    "Science Fiction": 1,
    "Horror": 2,
    "Romance": 3,
    "LitRPG/Progression Fantasy/Cultivation": 4,
    "Cozy Fantasy/Slice of Life": 5,
    "I have several stories, and I'd like to talk it over with my group!": 6,
    "several": 6,
}

SIZE_CODE = {"Pair": 2, "Small Group (3)": 3, "Small Group (4)": 4, "":5}

CONTENT_WARNINGS = [
    "Profanity",
    "Sexual Content",
    "Gore and Excessive Violence",
    "Traumatizing Content",    
]

SORT_WEIGHTS = {
    "group_veto": 10000,
    "seasonal" : 9900,
    "friend_req": 1000,
    "wordcount": 20,
    "cw_trauma": 18,
    "cw_sex": 18, # unused; sex currently uses the trauma weight
    "cw_gore": 18, # same as above
    "cw_swearing": 12,
    "group_size": 10,
    "size_pref": 5,
    "genre": 5,
}

QUESTION_CODES = {
    "Timestamp" : 'timestamp',
    "What's your COTEH Discord Username?" : 'name',
    'True or False: In critique groups, you should tell your other group members what they did wrong.' : 'quiz',
    'Have you been in critique houses before? ' : 'crit_history',
    'About how many words do you plan to bring to critique each week?' : 'words_wr',
    'What genre is your story? Please choose the one that BEST fits your story.' : 'genre_wr',
    'What Content Warnings does your story have? Please choose all that apply. (We do not currently host sexually explicit/smut critique. Sorry. Fade to Black should be fine.)' : 'cw_wr',
    'Week 1': 'wk_1',
    'Week 2': 'wk_2',
    'Week 3': 'wk_3',
    'Week 4': 'wk_4',
    'Have you completed a book before?' : 'book_done',
    'Would you prefer to work as a pair, a group of three, or four total people? Please choose all that apply.' : 'size_pref',
    "How many words can you commit to critiquing each week per Housemate? Try to line this up with what you're willing to bring yourself." : 'words_r',
    'What genres would you like to critique? Please choose ALL that apply.' : 'genre_r',
    'What content warnings are you not comfortable reading and critiquing? Please choose all that apply.  (We do not currently host sexually explicit/smut material; the option is included for completeness.)' : 'cw_veto',
    'Did you participate in crit groups last month?  If so, would you like to continue with the same group?' : 'prev_month',
    'Which crit group were you in last month?  (e.g. HippoHammer, BunnyBomb, etc.)' : 'prev_group',
    "If you didn't request to work with last month's group, is there anyone you want to be matched with?  (For best results, please only list their names, separated by commas.)" : 'match_request',
    'Is there anyone you would prefer NOT to work with?  (This question only ensures you are not matched together. If you want to report a problem, use the next question.  For best results, please only list their names, separated by commas.)' : 'match_veto',
    'Did you have any problem members this month? (e.g., no-shows, refused to give feedback, or overly abrasive/negative feedback.) Please list all members who apply so we can investigate.' : 'naughty_list',
    'Is there anyone you want to be matched with?  (For best results, please only list their names, separated by commas.)' : 'match_request',
}

# used for fuzzy matching on name fields for team requests and vetos
# values below 80 don't seem to work very well; max 99
NAME_TOLERANCE = 90

def extend_str(input, length):
    """
    adds whitespace to the end of a string
    until it reaches the specified length
    """
    gap_len = length - len(input)
    spacer = " " * gap_len if gap_len > 0 else ""
    return input + spacer


#######################################################
#                 class definitions                   #
#######################################################

class Person:
    def __init__(self, name: str, **kwargs) -> None:
        self.name = name.replace(" ", "")
        self.words_wr = kwargs["words_wr"]
        self.words_r = kwargs["words_r"]
        self.genre_wr = kwargs["genre_wr"]
        self.genre_r = kwargs["genre_r"]
        self.cw_wr = kwargs["cw_wr"]
        self.cw_veto = kwargs["cw_veto"] if "cw_veto" in kwargs else []
        self.size_pref = kwargs["size_pref"]
        self.match_pref = kwargs["match_pref"]
        if isinstance(self.match_pref, list):
            self.match_pref = " ".join(self.match_pref)
        self.match_veto = kwargs["match_veto"]
        if isinstance(self.match_veto, list):
            self.match_pref = " ".join(self.match_veto)
        self.seasonal = kwargs["seasonal"] if "seasonal" in kwargs else {}

    def word_dist(self, other):
        # exaggerating word output means 2 vs 1 weighs less than 3 vs 2
        my_diff = 1.1 * self.words_wr - other.words_r
        their_diff = 1.1 * other.words_wr - self.words_r
        word_diff = (max(my_diff, 0) + max(their_diff, 0)) ** 2
        word_diff *= WORDCOUNT_SCALE

        return word_diff 

    def friend_dist(self, other):
        my_friend = other.name in self.match_pref
        their_friend = self.name in other.match_pref
        return 1 if my_friend or their_friend else 0

    def veto_dist(self, other):
        my_foe = other.name in self.match_veto
        their_foe = self.name in other.match_veto
        return 1 if my_foe or their_foe else 0

    def genre_dist(self, other):
        score = 0
        if self.genre_wr not in other.genre_r:
            score += 1
        if other.genre_wr not in self.genre_r:
            score += 1
        return score

    def gen_dist(self, other):
        total = 0
        total += self.word_dist(other) * SORT_WEIGHTS["wordcount"]
        total += self.veto_dist(other) * SORT_WEIGHTS["group_veto"]
        total += self.genre_dist(other) * SORT_WEIGHTS["genre"]
        # we want friends to increase fit (e.g. make it lower)
        total -= self.friend_dist(other) * SORT_WEIGHTS["friend_req"]

        return max(total,0)

    def __repr__(self) -> str:
        return "\n".join(
            [
                self.name,
                str(self.words_wr),
                str(self.words_r),
                str(self.genre_wr),
                str(self.genre_r),
                str(self.cw_wr),
                str(self.cw_veto),
                str(self.size_pref),
                str(self.match_pref),
                str(self.match_veto),
                str(self.seasonal),
            ]
        )

class Group:
    def __init__(self, members=None, seasonal=False, reqs=None) -> None:
        self.members = members if members else []
        self.reqs = reqs if reqs else []
        if members:
            self.seasonal = members[0].seasonal
        else:
            self.seasonal = seasonal

    @property
    def size(self):
        return len(self.members)

    @property
    def max_size(self):
        return min([4] + [max(p.size_pref) for p in self.members])

    @property
    def max_read(self):
        return min([4] + [p.words_r for p in self.members])

    @property
    def avg_read(self):
        return sum([0] + [p.words_r for p in self.members])/self.size

    @property
    def max_output(self):
        return max([p.words_wr for p in self.members])

    @property
    def is_valid(self):
        size_constraint = self.size <= self.max_size
        words_constraint = self.max_output <= self.max_read
        return size_constraint and words_constraint

    @property
    def avg_output(self):
        return sum([p.words_wr for p in self.members]) / self.size

    @property
    def valid_seasonal(self) -> bool | None:
        return False not in [p.seasonal == self.seasonal for p in self.members]

    def __hash__(self) -> int:
        names = "".join(p.name for p in self.members)
        return reduce(lambda x,y : x+ord(y), names, 0)

    def permute_with(self, person):
        new_members = self.members.copy()
        new_members.append(person)
        return Group(members=new_members, seasonal=self.seasonal, reqs=self.reqs)

    def permute_with_members(self, members):
        return Group(members=members, seasonal=self.seasonal, reqs=self.reqs)

    def copy(self):
        return Group(members=self.members.copy(), seasonal=self.seasonal, reqs=self.reqs.copy())

    def merge_with_group(self, other):
        self.members = self.members + other.members.copy()
        return self
    
    def steal_least_compatible_member_from(self, other, virtual=False) -> tuple[int, float] | None:
        """
        Checks all possibilities where one group member is transfered from another group,
        then makes the transfer that best increases overall fit if one exists.
        The "virtual" flag prevents any actual transfer from occurring and just returns the result.
        """
        best_delta = 0
        result = None

        other_base_fit = other.fit()
        my_base_fit = self.fit()
        for p_idx in range(len(other.members)):
            other_prime = other.copy()
            other_orphan = other_prime.members.pop(p_idx)
            other_new_fit = other_prime.fit()

            me_prime = self.permute_with(other_orphan)
            me_new_fit = me_prime.fit()

            delta = other_new_fit + me_new_fit - my_base_fit - other_base_fit
            if delta < best_delta:
                best_delta = delta
                result = (p_idx, delta)

        if result != None:
            if virtual == False:
                self.members.append(other.members.pop(result[0]))
            return result
        else:
            return (None, 0)

    def fit(self):
        tmp = self.members.copy()
        score = 0
        discount = ((self.size) * (self.size - 1)) / 2

        # during seasonal events, shortcut out if group isn't all in the same event 
        if not self.valid_seasonal:
            return SORT_WEIGHTS["seasonal"]
        
        # ensure all friend requests are present
        group_names = [p.name for p in self.members]
        for name in self.reqs:
            if name not in group_names:
                score += SORT_WEIGHTS["friend_req"]

        # prefer groups of size 3
        size_penalty = abs(3 - self.size)
        score += SORT_WEIGHTS["group_size"] ** size_penalty

        cws = { cw : 0 for cw in CONTENT_WARNINGS }

        while tmp:
            p: Person = tmp.pop()
            score += 0 if self.size in p.size_pref else SORT_WEIGHTS["size_pref"] / self.size

            for cw in p.cw_wr:
                if cw in cws:
                    cws[cw] += 1

            for other in tmp:
                score += p.gen_dist(other) / discount

        # score for each cw people don't want to read
        for p in self.members:
            for cw in p.cw_veto:
                if cw in cws:
                    if cw == "Profanity":
                        score += cws[cw] * SORT_WEIGHTS["cw_swearing"]
                    else:
                        score += cws[cw] * SORT_WEIGHTS["cw_trauma"]

        return round(score, 2)

    def print_names(self) -> str:
        return ", ".join(p.name for p in self.members)

    def print_wc_table(self):
        output = f"Group with reqs {self.reqs}\n"
        longest_name_len = max([len(p.name) for p in self.members])
        for p in self.members:
            output += f"{extend_str(p.name, longest_name_len)} "
            output += f"wr {p.words_wr}, "
            output += f"r {p.words_r}, "
            output += f"s {max(p.size_pref)}"
            output += "\n"
        return output

    def print_cw_table(self, force_longest=0):
        longest_name_len = force_longest if force_longest else max([len(p.name) for p in self.members])
        output = f"{extend_str("NAME", longest_name_len)} PROF SEX  GORE TRAU OTHER\n"
        for p in self.members:
            output += f"{extend_str(p.name, longest_name_len)} "
            for cw in CONTENT_WARNINGS:
                if cw in p.cw_wr:
                    cw_code = "√"
                elif cw in p.cw_veto:
                    cw_code = "!"
                else:
                    cw_code = " "
                output += extend_str(cw_code, 5)
            output += "\n"
        return output

    def __repr__(self) -> str:
        return f"<<Group of size {self.size} with members {[mem.name for mem in self.members]} and fit score {self.fit()}>>"

class Model:
    def __init__(self, groups=None, users=None) -> None:
        self.groups = groups if groups else []
        self.users = users if users else []

    @property
    def group_count(self):
        return len(self.groups)

    @property
    def total_fit(self) -> float:
        return round(sum([g.fit() for g in self.groups]),2)

    @property
    def avg_fit(self) -> float:
        return round(self.total_fit / len(self.groups), 2)

    @property
    def median_fit(self) -> float:
        fits = [g.fit() for g in self.groups]
        fits.sort()
        return fits[len(fits)//2]

    def sort_users(self, func) -> None:
        self.users.sort(key=func)

    def get_req_clusters(self) -> list:
        # step one: each person with friend requests is a list
        req_users = [p for p in self.users if p.match_pref]

        clusters = []
        for user in req_users:
            friends = set(p for p in self.users if p.name in user.match_pref and user.name not in p.match_veto)

            if friends:
                friends.add(user)
                clusters.append(friends)

        # step two: combine lists with overlap; may produce overlarge groups
        output = []

        while clusters:
            curr = clusters.pop(0)

            idx = 0
            while idx < len(clusters):

                if curr & clusters[idx]:
                    other = clusters.pop(idx)
                    curr = curr | other
                    idx = 0
                else:
                    idx += 1
            output.append(Group(list(curr), reqs=[p.name for p in curr]))
        return output

    def make_groups(self, sort="default", premades=None, fill_incomplete_premades=True) -> None:
        """
        If the model has a list of users, assigns them to groups.
        :premades: is a list of premade groups. People in those
                   groups will not be considered by the algorithm.
        :fill_incomplete_premades: boolean; if True, the model will
                   add users to premade groups with size 2 or less.
        """
        assert len(self.users) > 0

        # grouping by size preference clumps people with lower prefs
        # otherwise they scatter and block all larger groups
        self.sort_users(lambda p: max(p.size_pref))
        self.sort_users(lambda p: p.words_r)
        self.sort_users(lambda p: p.words_wr)

        if sort == "seasonal":
            self.sort_users(lambda p: p.seasonal)

        self.groups = premades.copy() if premades else []
        users = self.users.copy()

        # if premades, remove assigned members from list
        if premades:
            member_lists = [g.members for g in premades]
            preassigned = reduce(lambda x, y: x + y, member_lists, [])
            users = [p for p in users if p not in preassigned]

            # steal unassigned users to fill out incompletes
            if fill_incomplete_premades:
                for g_idx in range(self.group_count):
                    g = self.groups[g_idx]
                    if g.size < g.max_size:
                        tmp = [(g.permute_with(users[idx]), idx) for idx in range(len(users)) if g.permute_with(users[idx]).is_valid]
                        tmp.sort(key=lambda x: x[0].fit())
                        new_g, idx = tmp[0] if tmp else (g, None)
                        self.groups[g_idx] = new_g
                        if not (idx is None):
                            users.pop(idx)

        curr_group : Group = Group([])

        while users:
            p = users.pop(0)

            if curr_group.size < max(p.size_pref) and curr_group.size < curr_group.max_size:
                curr_group = curr_group.permute_with(p)
            else:
                self.groups.append(curr_group)
                curr_group = Group([p])

        self.groups.append(curr_group)

        self.balance_group_weights()
        return

    def balance_group_weights(self) -> None:
        """
        once groups have been assigned, does a pass to see
        if swapping members between adjacent groups improves
        fit.
        """
        assert self.group_count > 0
        self.groups.sort(key=lambda g: g.avg_read)
        self.groups.sort(key=lambda g: g.avg_output)
        self.groups.sort(key=lambda g: g.max_output)

        for g_idx in range(1, len(self.groups)-1):
            self.get_local_group_minima(g_idx)

        self.flatten_group_variance()
        self.reassign_singletons()

        for g_idx in range(1, len(self.groups)-1):
            self.get_local_group_minima(g_idx)

    def reassign_singletons(self):
        """
        remove all groups of 1 and shove the members elsewhere
        """
        for g_idx in range(self.group_count-1, 0, -1):
            if self.groups[g_idx].size == 1:
                orphan = self.groups.pop(g_idx).members.pop()
                self.adopt_orphan(orphan)

    def flatten_group_variance(self):
        """
        if there are groups of 2, steal members from groups of 4 to even them out
        """
        size_2 = [g_idx for g_idx in range(len(self.groups)) if self.groups[g_idx].size == 2]
        size_2.sort(key=lambda x : self.groups[x].fit(), reverse=True)

        while size_2:

            dest_group_idx = size_2.pop(0)
            dest_group : Group = self.groups[dest_group_idx]
            best_delta = 0
            best_origin = None

            size_4 = [g_idx for g_idx in range(len(self.groups)) if self.groups[g_idx].size == 4]
            while size_4:
                
                origin_group_idx = size_4.pop(0)
                origin_group = self.groups[origin_group_idx]

                _, delta = dest_group.steal_least_compatible_member_from(origin_group, virtual=True)
                if delta < best_delta:
                    best_delta = delta
                    best_origin = self.groups[origin_group_idx]

            if best_origin:
                dest_group.steal_least_compatible_member_from(best_origin)

    def get_local_group_minima(self, g_idx) -> tuple[Group, Group, Group]:
        g1, g2, g3 = self.groups[g_idx-1:g_idx+2]
        
        tmp1_1, tmp1_2, _ = self.osmose_groups(g1, g2)
        tmp1_2, tmp1_3, _ = self.osmose_groups(tmp1_2, g3)
        tmp1_fit = tmp1_1.fit() + tmp1_2.fit() + tmp1_3.fit()

        tmp2_2, tmp2_3, _ = self.osmose_groups(g2, g3)
        tmp2_1, tmp2_2, _ = self.osmose_groups(g1, tmp2_2)
        tmp2_fit = tmp2_1.fit() + tmp2_2.fit() + tmp2_3.fit()

        if tmp1_fit < tmp2_fit:
            self.groups[g_idx-1] = tmp1_1
            self.groups[g_idx  ] = tmp1_2
            self.groups[g_idx+1] = tmp1_3
        else:
            self.groups[g_idx-1] = tmp2_1
            self.groups[g_idx  ] = tmp2_2
            self.groups[g_idx+1] = tmp2_3

    def adopt_orphan(self, person:Person) -> tuple[int, float]:
        """
        we try inserting the given person into each group, then return a tuple containing:
            - the group index where they went
            - the fit delta for placing them there
        """
        deltas = [ g_idx for g_idx in range(self.group_count) if self.groups[g_idx].size < 4 ]

        for g_idx in range(len(deltas)):
            base_fit = self.groups[g_idx].fit()
            new_fit = self.groups[g_idx].permute_with(person).fit()
            delta = new_fit - base_fit
            deltas[g_idx] = (g_idx, delta)

        deltas.sort(key=lambda x: x[1])
        g_idx, fit = deltas[0]
        self.groups[g_idx] = self.groups[g_idx].permute_with(person)
        return (g_idx, fit)

    def osmose_groups(self, g1:Group, g2:Group) -> tuple[Group, Group, float]:
        """
        swaps each possible pair of group members to determine the regime
        with the best fit
        """

        base_fit = g1.fit() + g2.fit()
        best_fit = base_fit
        result = (g1, g2, best_fit)

        for idx1 in range(g1.size):
            for idx2 in range(g2.size):
                tmp1 = g1.members.copy()
                tmp2 = g2.members.copy()    

                swap1 = tmp1.pop(idx1)
                swap2 = tmp2.pop(idx2)

                tmp_group1 = g1.permute_with_members(tmp1 + [swap2])
                tmp_group2 = g2.permute_with_members(tmp2 + [swap1])

                curr_fit = tmp_group1.fit() + tmp_group2.fit()
                if curr_fit < best_fit:
                    best_fit = curr_fit
                    result = (tmp_group1, tmp_group2, best_fit)

        if best_fit == base_fit:
            return (g1, g2, base_fit)
        else:
            return self.osmose_groups(result[0], result[1])

    def copy(self):
        new_groups = [g.copy() for g in self.groups]
        return Model(groups=new_groups, users=self.users.copy())

    def print_groups(self) -> str:
        return "\n".join([str(g) for g in self.groups])

    def print_fit(self) -> str:
        return f"total fit {self.total_fit}, avg fit {self.avg_fit} | {self.median_fit}"

    def __repr__(self) -> str:
        return f"<<Model with {len(self.users)} users in {self.group_count} groups; {self.print_fit()}>>"

#######################################################
#                  loose functions                    #
#######################################################

def translate_fb_pref(fb_str) -> str:
    return (
        "Writer" if "Writer" in fb_str else "Reader" if "Reader" in fb_str else "Hype"
    )

def clean_team_reqs(users) -> None:
    """
    replace team requests and vetos with the names that
    are actually being used
    """
    if len(users) == 0: return

    name_list = [p.name for p in users]

    for p in users:
        match_pref = p.match_pref
        if match_pref:
            p.match_pref = [name for name in name_list if fuzz.partial_ratio(name, match_pref) > NAME_TOLERANCE]

        match_veto = p.match_veto
        if match_veto:
            p.match_veto = [name for name in name_list if fuzz.partial_ratio(name, match_veto) > NAME_TOLERANCE]


#######################################################
#                   begin script                      #
#######################################################

users = []

with open("source.csv", "r", encoding="utf-8") as f:
    # input = "\r".join([line for line in file])
    data = list(csv.reader(f))

    header = [question.replace("\n","") for question in data[0]]
    input = data[1:]

    # print(header)
    # print(input)

# header = clean_data(header).split(",")
columns = { QUESTION_CODES[question]: idx 
           for (idx, question) in zip(range(len(header)), header) }

for response in input:

    # response = clean_data(response).strip().split(",")

    name = response[columns['name']]

    # how much output will you bring?
    words_wr = WORDCOUNT_CODE[response[columns['words_wr']]]

    # what genre do you write?
    genre_wr = response[columns['genre_wr']].split("& ")

    # what content warnings apply to your story?
    cw_wr = [
        GENRE_CODE[cw] if cw in GENRE_CODE else cw for cw in response[columns['cw_wr']].split("& ")
    ]

    # chapter links
    chapters = [response[columns['wk_1']], 
                response[columns['wk_2']], 
                response[columns['wk_3']],
                response[columns['wk_4']]]

    # what size group are you okay with?
    size_pref = [SIZE_CODE[pref] for pref in response[columns['size_pref']].split(", ")]

    # how many words can you commit to critiquing each week?
    words_r = WORDCOUNT_CODE[response[columns['words_r']]]

    # what genres will you be bringing to group?
    genre_r = [
        GENRE_CODE[genre] if genre in GENRE_CODE else genre
        for genre in response[columns['genre_r']].split("& ")
    ]

    # which content warnings do you want to avoid?
    cw_veto = (
        []
        if response[columns['cw_veto']] == "I'll read anything!"
        else [
            GENRE_CODE[cw] if cw in GENRE_CODE else cw
            for cw in response[columns['cw_veto']].split("& ")
        ]
    )

    # who would you like to be with?
    match_pref = response[columns['match_request']]

    # who do you want to avoid?
    match_veto = response[columns['match_veto']]

    # # do you want to be in a contest-focused group?
    # contest = response[-1] == "Yes"

    user = Person(
        name,
        words_wr=words_wr,
        genre_wr=genre_wr,
        cw_wr=cw_wr,
        size_pref=size_pref,
        words_r=words_r,
        genre_r=genre_r,
        cw_veto=cw_veto,
        match_pref=match_pref,
        match_veto=match_veto,
        # seasonal=contest,
    )

    users.append(user)


clean_team_reqs(users)

for user in users:
    print(user.name)
# print(users)

# contest = [p for p in users if p.seasonal]
# non_contest = [p for p in users if not p.seasonal]

# m1 = Model(users=contest.copy())
# m2 = Model(users=non_contest.copy())

m = Model(users=users.copy())
# for m in [m1, m2]:

longest_name = max(len(p.name) for p in m.users)
reqs = m.get_req_clusters()
m.make_groups(premades=reqs)

print(m)
print(m.print_groups())

for g in m.groups:
    print("----------------")
    print(g.print_wc_table())
    print(g.print_cw_table(longest_name))
print()