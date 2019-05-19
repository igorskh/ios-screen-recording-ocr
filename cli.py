#!/usr/bin/python3
__author__ = "Igor Kim"
__credits__ = ["Igor Kim"]
__maintainer__ = "Igor Kim"
__email__ = "igor.skh@gmail.com"
__status__ = "Development"
__date__ = "05/2019"
__license__ = "MIT"

import os, sys, argparse, time, math, logging
import pandas as pd
import multiprocessing as mp

import recognizer, consts, cv_helpers

def mp_job(fnames, q, padding=0, scaling_factor=1, debug=False):
    for f in fnames:
        has_value, has_none, result = recognizer.process_one_image(f, None, padding, scaling_factor, debug)
        q.put(result)

def append_result(res, data):
    for k in res:
        data[k].append(res[k])

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, help="Input image path")
ap.add_argument("-o", "--output", type=str, help="Output image path")
ap.add_argument("-c", "--output-csv", type=str, default="%s/pandas.csv"%consts.DEBUG_FOLDER, help="Path to output CSV file")
ap.add_argument("-d", "--debug", type=bool, default=False, help="Debug mode")
ap.add_argument("-v", "--video", type=bool, default=False, help="Extract video frames")
ap.add_argument("-p", "--padding", type=int, default=0, help="Fixed padding")
ap.add_argument("-n", "--n-proc", type=int, default=2, help="Number of cores for multiprocessing")
ap.add_argument("-l", "--limit", type=int, default=None, help="Number of file to process")
ap.add_argument("-s", "--scaling-factor", type=str, default="1", help="Scaling factor for Tesseract")
ap.add_argument("-t", "--video-tolerance", type=float, default=.98, help="Tolerance for video import")
args = vars(ap.parse_args())

handlers = [logging.StreamHandler()]
logging.basicConfig(
    level=logging.DEBUG if args["debug"] else logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=handlers)
logger = logging.getLogger('')

args["scaling_factor"] = list(map(int, args["scaling_factor"].split(",")))
for f in consts.FOLDERS:
    if not os.path.exists("build"):
        os.makedirs("build")

if not os.path.exists(args["input"]):
    logger.critical("File or folder %s not found"%args["input"])
    sys.exit()

if args["video"]:
    if not os.path.exists(args["output"]):
        os.makedirs(args["output"])

    tolerance = args["video_tolerance"]
    n_proc = args["n_proc"]
    n_frames, fps = cv_helpers.get_video_n_frames(args["input"])

    if fps == 0:
        fps = 1
        n_proc = 1
        logger.warning("Could not detect video FPS, fallback to 1 thread")
        n_jobs = n_frames
        procs = [mp.Process(target=cv_helpers.get_video_frames, 
            args=(args["input"], args["output"], None, None, tolerance)) for i in range(n_proc)]
    else:
        frames = list(range(0, n_frames, fps))
        _, middle_frame = cv_helpers.get_video_frame(args["input"], frames[math.floor(len(frames)/2)])
        n_jobs = len(frames)
        p_index = list(range(n_jobs))
        chunk_size = int(math.ceil(n_jobs/n_proc))
        res = [frames[i:i+chunk_size] for i in range(0, n_jobs, chunk_size)]
        n_proc = len(res)

        procs = [mp.Process(target=cv_helpers.get_video_frames, 
            args=(args["input"], args["output"], res[i], middle_frame, tolerance)) for i in range(n_proc)]
    
    start = time.time()
    for p in procs:
        p.start()
    logger.info("Extracting %d frames in %d thread(s)"%(n_jobs, n_proc))
    for p in procs:
        p.join()
    logger.info("Extracted %d in %.2f s"%(n_jobs, time.time() - start))
    sys.exit()

data = {}
for k in consts.EXPECTED_KEYS:
    data[k] = []
data["file"] = []
if os.path.isfile(args["input"]):
    logger.info("Parsing file %s"%args["input"])
    has_value, has_none, res = recognizer.process_one_image(args["input"], args["output"], 
        args["padding"], args["scaling_factor"], debug=args["debug"])
    append_result(res, data)
else:
    folder = args["input"]
    logger.info("Parsing folder %s"%folder)

    frames_fn = []
    idx = 0
    for f in os.listdir(folder):
        if f.endswith(".png"):
            frames_fn.append(os.path.join(folder, f))
        idx += 1
        if args["limit"] is not None and args["limit"] > 0 and idx == args["limit"]:
            break

    q = mp.Queue()
    n_proc = args["n_proc"]
    n_jobs = len(frames_fn)
    p_index = list(range(n_jobs))
    chunk_size = int(math.ceil(n_jobs/n_proc))
    res = [frames_fn[i:i+chunk_size] for i in range(0, n_jobs, chunk_size)]
    n_proc = len(res)
    start = time.time()

    procs = [mp.Process(target=mp_job, args=(res[i], q, args["padding"], 
        args["scaling_factor"], args["debug"])) for i in range(n_proc)]
    for p in procs:
        p.start()

    logger.info("Starting jobs in %d thread(s)"%n_proc)
    for p in procs:
        p.join()

    while not q.empty():
        append_result(q.get(), data)
    logger.info("Parsed %d in %.2f s"%(n_jobs, time.time() - start))

df = pd.DataFrame(data=data)
df = df.groupby("timestamp").first().sort_values(by=["timestamp"]).reset_index()
df.to_csv(args["output_csv"])