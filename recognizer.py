__author__ = "Igor Kim"
__credits__ = ["Igor Kim"]
__maintainer__ = "Igor Kim"
__email__ = "igor.skh@gmail.com"
__status__ = "Development"
__date__ = "05/2019"
__license__ = "MIT"

import cv2, logging

import consts, cv_helpers, text_helpers

logger = logging.getLogger('')
def detect_keys(res, expected_keys=consts.EXPECTED_KEYS):
    if len(res) != len(expected_keys):
        return None
    found_keys = []
    for r in res:
        found = False
        for k in expected_keys:
            if k in found_keys:
                continue
            val = r
            if expected_keys[k]["corr"] == consts.CHECK_DISTANCE:
                val = text_helpers.check_distance(val, k)
            elif expected_keys[k]["corr"] == consts.CHECK_REPLACE:
                val = text_helpers.check_replace(val, consts.REPLACE_RULES)
            if val == k:
                found_keys.append(val)
                found = True
        if len(found_keys) == len(expected_keys) or not found:
            break
    return found_keys

def detect_values(res, keys, result, expected_keys=consts.EXPECTED_KEYS):
    if len(res) != len(expected_keys):
        return None

    has_none = False
    for i in range(len(res)):
        val = None
        try:
            val = expected_keys[keys[i]]["map"](res[i])
        except ValueError:
            logger.warning("Could not map value %s to %s"%(res[i], keys[i]))
            has_none = True
        if "range" in expected_keys[keys[i]] and  not has_none and not text_helpers.is_val_in_range(val, expected_keys[keys[i]]["range"]):
            logger.warning("Value %s of type %s is out of range"%(res[i], keys[i]))
            val = None
            has_none = True
        if result[keys[i]] is None:
            result[keys[i]] = val
    return result, has_none

def process_one_image(input_path, output_path=None, padding=0, scaling_factor=1, debug=False):
    original_image = cv2.imread(input_path)
    processed_image = cv_helpers.preprocess_image(original_image)
    if debug and output_path is not None:
        cv2.imwrite(output_path + "_processed.png", processed_image)
    cntrs = cv_helpers.find_countours(processed_image)
    output_image = cv_helpers.draw_countrous(original_image, cntrs)
    if debug and output_path is not None:
        cv2.imwrite(output_path + "_output.png", output_image)
    rois = cv_helpers.get_rois(original_image, cntrs)

    original_width = original_image.shape[:2][1]

    result = {}
    for k in consts.EXPECTED_KEYS:
        result[k] = None
    result["file"] = input_path

    first_key = list(consts.EXPECTED_KEYS.keys())[0]
    # sometimes things work with different scaling of the word
    for i in range(len(scaling_factor)):
        s = scaling_factor[i]
        res = cv_helpers.recognise_rois(rois, padding, s, debug)

        potential_keys = []
        potential_values = []
        # get keys from the left half and values from the right
        for r,p in res:
            if p[2] < original_width/4:
                potential_keys.append(r)
            if p[3] > original_width/2:
                potential_values.append(r)

        if len(res)%2 != 0:
            logger.warning("Cannot process - odd number of values [%s] scaling factor [%d]"%(input_path, s))
            continue
        keys = detect_keys(potential_keys)
        if keys is None or len(keys) != len(consts.EXPECTED_KEYS):
            logger.warning("Keys not found [%s] scaling factor [%d]"%(input_path, s))
            continue
        values, has_none = detect_values(potential_values, keys, result, consts.EXPECTED_KEYS)
        if values is None:
            logger.warning("Values not found [%s] scaling factor [%d]"%(input_path, s))
            continue
        if has_none:
            logger.warning("Some values are None [%s] scaling factor [%d]"%(input_path, s))
            continue
        break
    logger.debug(keys)
    logger.debug(values)
    
    has_none = False
    has_value = False
    for r in result:
        if result[r] is None:
            has_none = True
        else:
            has_value = True
    return has_value, has_none, result
