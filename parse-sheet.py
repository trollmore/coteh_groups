from fuzzywuzzy import fuzz
from fuzzywuzzy import process
from functools import reduce

WORDCOUNT_CODE = {
    "Up to 2&000": 0,
    "Between 2&000 and 4&000": 1,
    "Between 4&000 and 6&000": 2,
    "Over 6&000": 3,
    "I don't care& I just like pre-reading stuff!": 4,
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
    "What's your approximate age?" : 'is_minor',
    "What are you mainly looking for feedback on?" : 'fb_wanted',
    "About how many words do you plan to bring to critique each week?" : 'words_wr',
    "What genre is your story? Please choose the one that BEST fits your story." : 'genre_wr',
    "What Content Warnings does your story have? Please choose all that apply. (We do not currently host sexually explicit/smut critique. Sorry. Fade to Black should be fine.)" : 'cw_wr',      
    "We're asking people to be sure they have four chapters they'd like to have critiqued before they sign up in order to guarantee that they'll have material to bring to the later weeks of critique groups. Please include a link to a Google Doc with your four chapters." : 'chapters',
    "Would you prefer to work as a pair& a group of three& or four total people? Please choose all that apply." : 'size_pref',
    "What feedback do you feel comfortable giving?" : 'fb_provide',
    "How many words can you commit to critiquing each week per Housemate? Try to line this up with what you're willing to bring yourself." : 'words_r',
    "What genres would you like to critique? Please choose ALL that apply." : 'genre_r',
    "What Content Warnings are you not comfortable reading and critiquing? Please choose all that apply.  (We do not currently host sexually explicit/smut critique. Sorry. Fade to Black should be fine.)" : 'cw_veto',
    "If you were on a team and would like to work with them again& please list the members of your team and your team name. This question is optional." : 'team_pref',
    "Is there anyone you would prefer NOT to work with again?" : 'team_veto',
    "Did you have any problem members (no-shows& refused to give feedback& or overly abrasive/negative feedback)? Please list all members who apply so we can investigate." : 'naughty_list',
    "I verify that I've read the Manuscript Critiques infographic above.\n" : 'verify',
    'Have you been in critique houses before? ' : 'prev_crit',
    'Have you completed a book before? (This question is not disqualifying& just for research and team balance purposes.)' : 'book_done'
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
        if isinstance(self.team_pref, list):
            self.team_pref = " ".join(self.team_pref)
        self.team_veto = kwargs["team_veto"]
        if isinstance(self.team_veto, list):
            self.team_pref = " ".join(self.team_veto)
        self.seasonal = kwargs["seasonal"] if "seasonal" in kwargs else {}

    def word_dist(self, other):
        # exaggerating word output means 2 vs 1 weighs less than 3 vs 2
        my_diff = 1.1 * self.words_wr - other.words_r
        their_diff = 1.1 * other.words_wr - self.words_r
        word_diff = (max(my_diff, 0) + max(their_diff, 0)) ** 2
        word_diff *= WORDCOUNT_SCALE

        return word_diff 

    def friend_dist(self, other):
        my_friend = other.name in self.team_pref
        their_friend = self.name in other.team_pref
        return 1 if my_friend or their_friend else 0

    def veto_dist(self, other):
        my_foe = other.name in self.team_veto
        their_foe = self.name in other.team_veto
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
                str(self.team_pref),
                str(self.team_veto),
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
        req_users = [p for p in self.users if p.team_pref]

        clusters = []
        for user in req_users:
            friends = set(p for p in self.users if p.name in user.team_pref and user.name not in p.team_veto)

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

def clean_data(raw_data: str) -> str:
    """
    cleans response data for a user submission.

    Note that carriage return characters in response data will read
    as if each one is a separate submission and should be dealt with
    before this point.
    """

    # checksum for quotes
    if raw_data.count('"') % 2 != 0:
        criminal = raw_data.split(",")[1]
        raise ValueError(
            f'Malformed input: odd number of quote characters ( " ) in response data for {criminal}.'
        )

    # replace commas within responses with &
    result: str = raw_data
    result = result.replace("&", " ") # first get rid of existing '&' characters

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
        team_pref = p.team_pref
        if team_pref:
            p.team_pref = [name for name in name_list if fuzz.partial_ratio(name, team_pref) > NAME_TOLERANCE]

        team_veto = p.team_veto
        if team_veto:
            p.team_veto = [name for name in name_list if fuzz.partial_ratio(name, team_veto) > NAME_TOLERANCE]

def strip_newlines_from_responses(text:str) -> str:
    """
    eliminates all newline characters from user submissions,
    leaving only the ones from the csv
    """

    # "I've read it!\r" only occurs at the end of the submission
    return text.replace("I've read it!\r", "\\newline").replace("\r", " ").replace("\\newline", "\r")


#######################################################
#                   begin script                      #
#######################################################

users = []

with open("source.csv", "r", encoding="utf-8") as f:
    # input = "\r".join([line for line in file])
    header = f.readline()
    input = f.read().split("I've read it!")

header = clean_data(header).split(",")
columns = { QUESTION_CODES[question]: idx for (idx, question) in zip(range(len(header)), header) }

for response in input[:-1]:

    cleaned = clean_data(response).strip().split(",")

    # ######## uncomment this block in case the questionnaire
    # ######## changes and we need to change index values again
    # idx = 0
    # for d in cleaned:
    #     print(idx, d)
    #     idx += 1
    # break

    name = cleaned[columns['name']]

    # age
    is_minor = "Under" in cleaned[columns['is_minor']]

    # what kind of feedback are you looking for?
    fb_wanted = translate_fb_pref(cleaned[columns['fb_wanted']])

    # how much output will you bring?
    words_wr = WORDCOUNT_CODE[cleaned[columns['words_wr']]]

    # what genre do you write?
    genre_wr = cleaned[columns['genre_wr']].split("& ")

    # what content warnings apply to your story?
    cw_wr = [
        GENRE_CODE[cw] if cw in GENRE_CODE else cw for cw in cleaned[columns['cw_wr']].split("& ")
    ]

    # chapter links
    chapters = cleaned[columns['chapters']]

    # what size group are you okay with?
    size_pref = [SIZE_CODE[pref] for pref in cleaned[columns['size_pref']].split("& ")]

    # what kind of feedback are you okay giving?
    fb_provide = [translate_fb_pref(cleaned[columns['fb_provide']])]

    # how many words can you commit to critiquing each week?
    words_r = WORDCOUNT_CODE[cleaned[columns['words_r']]]

    # what genres will you be bringing to group?
    genre_r = [
        GENRE_CODE[genre] if genre in GENRE_CODE else genre
        for genre in cleaned[columns['genre_r']].split("& ")
    ]

    # which content warnings do you want to avoid?
    cw_veto = (
        []
        if cleaned[columns['cw_veto']] == "I'll read anything!"
        else [
            GENRE_CODE[cw] if cw in GENRE_CODE else cw
            for cw in cleaned[columns['cw_veto']].split("& ")
        ]
    )

    # who would you like to be with?
    team_pref = cleaned[columns['team_pref']]

    # who do you want to avoid?
    team_veto = cleaned[columns['team_veto']]

    # # do you want to be in a contest-focused group?
    # contest = cleaned[-1] == "Yes"

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
        # seasonal=contest,
    )

    users.append(user)

clean_team_reqs(users)

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
