from pyjarowinkler import distance


asd = ['OrderedDict'
,'BasicCartesianConstraint'
,'UpdateGodMap'
,'GiskardException'
,'FrameInput'
,'ExternalCollisionAvoidance'
,'PoseStampedInput'
,'SoftConstraint'
,'CartesianPosition'
,'CartesianOrientation'
,'JointPositionList'
,'PointStampedInput'
,'Vector3Input'
,'Constraint'
,'GravityJoint'
,'Point3Input'
,'CartesianPose'
,'Vector3StampedInput'
,'CartesianPositionY'
,'CartesianPositionX'
,'SelfCollisionAvoidance'
,'AlignPlanes'
,'Pointing'
,'JointPosition'
,'TranslationInput'
,'Vector3Stamped'
,'CartesianOrientationSlerp']



def jaro_winkler_distance(s1, s2):
    length_s1 = len(s1)
    length_s2 = len(s2)
    num_matches = 0
    num_transposition = 0

    max_d = (max(length_s1, length_s2) // 2) - 1

    i = 0
    last_match_pos = 0

    while i < min(length_s1, length_s2 + max_d):
        j = max(0, i-max_d)
        while j <= i+max_d and j < length_s2:
            if (s1[i] == s2[j]):
                num_matches += 1;
                if (j < last_match_pos):
                    num_transposition += 2
                last_match_pos = j #max(last_match_pos, j)
                s2 = s2[:j] + '\1' + s2[j+1:]
                #s2[j] = 0
                break
            j += 1
        i += 1
    num_transposition =  num_transposition / 2.0
    if num_matches <= 0:
        return 0.0
    return (num_matches / float(length_s1) + num_matches / float(length_s2) + (num_matches - num_transposition) / float(num_matches)) / 3


def jaro_winkler_distance2(s1, s2):
    length_s1 = len(s1)
    length_s2 = len(s2)
    num_matches = 0
    num_transposition = 0

    max_d = (max(length_s1, length_s2) / 2) - 1

    i = 0
    last_match_pos = 0
    transpositions = []

    while i < min(length_s1, length_s2 + max_d):
        if s1[i] == s2[i]:
            if i in transpositions:
                num_transposition -= 1
            num_matches += 1
            s2 = s2[:i] + '\0' + s2[i + 1:]
        else:
            j = max(0, i-max_d)
            while j < i+max_d and j < length_s2:
                if (s1[i] == s2[j]):
                    num_matches += 1;
                    num_transposition += 1
                    transpositions.append(j)
                j += 1
        i += 1
    return (num_matches / float(length_s1) + num_matches / float(length_s2) + (num_matches - num_transposition) / float(num_matches)) / 3



import difflib


matches = difflib.get_close_matches("cart", asd,10, cutoff=0.3)
print(matches)


s1 = "joint"
for s2 in asd:
    s = difflib.SequenceMatcher(None, s1, s2)
    ratio = s.ratio()
    if ratio >= 0.3:
        print(s1 + " " + s2 + " " + str(ratio))
#s1 = "hqti"
#s2 = "tqhsfd"
#a = jaro_winkler_distance(s1, s2)
##print distance.get_jaro_distance(s1, s2)
#print(a)













