__author__ = "Igor Kim"
__credits__ = ["Igor Kim"]
__maintainer__ = "Igor Kim"
__email__ = "igor.skh@gmail.com"
__status__ = "Development"
__date__ = "05/2019"
__license__ = "MIT"

def minimum_edit_distance(s1,s2):
    """Calculates Levenshtein distance for strings s1 and s2
    Source: https://rosettacode.org/wiki/Levenshtein_distance#Python
    
    Arguments:
        s1 {str} -- string 1
        s2 {str} -- string 2
    
    Returns:
        int -- distance
    """
    if len(s1) > len(s2):
        s1,s2 = s2,s1
    distances = range(len(s1) + 1)
    for index2,char2 in enumerate(s2):
        newDistances = [index2+1]
        for index1,char1 in enumerate(s1):
            if char1 == char2:
                newDistances.append(distances[index1])
            else:
                newDistances.append(1 + min((distances[index1],
                                             distances[index1+1],
                                             newDistances[-1])))
        distances = newDistances
    return distances[-1]

def is_val_in_range(val, range):
    """Checks if val falls in range
    
    Arguments:
        val {float} -- value to check
        range {tuple} -- range of two values
    
    Returns:
        bool -- True if val in range
    """
    return val > range[0] and val < range[1]
    
def check_distance(val, val_expected, threshold=3):
    """Checks the Levenshtein distance with the threshold and returns
    expected string if the distance below or equal the threshold
    
    Arguments:
        val {str} -- string 1 - to compare
        val_expected {str} -- string 2 - expected
    
    Keyword Arguments:
        threshold {int} -- threshold for the distance (default: {3})
    
    Returns:
        str -- string 1 if distance above the threshold, string 2 otherwise
    """
    return val_expected if minimum_edit_distance(val, val_expected)<=threshold else val

def check_replace(val, replace_rules):
    """Replaces string in val by rules in dictionary replace_rules
    
    For example:  
    REPLACE_RULES = {
        "1,-1": ["i", "[", "]", "l", "7", "?", "t"],
        "q,": ["qg","qq","gg","gq"]
    }
    Arguments:
        val {str} -- input string
        replace_rules {dict} -- rules to replace
    
    Returns:
        str -- output string with replacements
    """
    val = str(val).lower()
    for replace_to in replace_rules:
        replace_to_ = replace_to.split(",")
        if len(replace_to_[1]) == 0:
            replace_to_[1] = None
        else:
            replace_to_[1] = int(replace_to_[1])
        for replace_from in replace_rules[replace_to]:
            if replace_to_[1] is None:
                val = val.replace(replace_from, replace_to_[0])
                continue
            if replace_to_[1] < 0:
                val = val[:replace_to_[1]] + val[replace_to_[1]].replace(replace_from, replace_to_[0])
            else:
                val = val[replace_to_[1]].replace(replace_from, replace_to_[0]) + val[replace_to_[1]+1:]
    return val