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

DEFAULT_PADDING = 20

def mp_job(fnames, q, padding=0, scaling_factor=1, debug=False):
    for f in fnames:
        has_value, has_none, result = recognizer.process_one_image(f, None, padding, scaling_factor, debug)
        q.put(result)

def append_result(res, data):
    for k in res:
        data[k].append(res[k])

def extract_frames(input_path, output_folder, n_proc):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    n_frames, fps = cv_helpers.get_video_n_frames(input_path)
    if fps == 0:
        fps = 1
        n_proc = 1
        logger.warning("Could not detect video FPS, fallback to 1 thread")
        n_jobs = n_frames
        procs = [mp.Process(target=cv_helpers.get_video_frames, 
            args=(input_path, output_folder, None, None)) for i in range(n_proc)]
    else:
        frames = list(range(0, n_frames, fps))
        _, middle_frame = cv_helpers.get_video_frame(input_path, frames[math.floor(len(frames)/2)])
        n_jobs = len(frames)
        p_index = list(range(n_jobs))
        chunk_size = int(math.ceil(n_jobs/n_proc))
        res = [frames[i:i+chunk_size] for i in range(0, n_jobs, chunk_size)]
        n_proc = len(res)

        procs = [mp.Process(target=cv_helpers.get_video_frames, 
            args=(input_path, output_folder, res[i], middle_frame)) for i in range(n_proc)]

    start = time.time()
    for p in procs:
        p.start()
    logger.info("Extracting %d frames in %d thread(s)"%(n_jobs, n_proc))
    for p in procs:
        p.join()
    logger.info("Extracted %d in %.2f s"%(n_jobs, time.time() - start))

def parse_folder(folder, output_csv, n_proc, scaling_factor):
    data = {}
    for k in consts.EXPECTED_KEYS:
        data[k] = []
    data["file"] = []

    logger.info("Parsing folder %s"%folder)
    frames_fn = []
    for f in os.listdir(folder):
        if f.endswith(".png"):
            frames_fn.append(os.path.join(folder, f))

    q = mp.Queue()
    n_jobs = len(frames_fn)
    p_index = list(range(n_jobs))
    chunk_size = int(math.ceil(n_jobs/n_proc))
    res = [frames_fn[i:i+chunk_size] for i in range(0, n_jobs, chunk_size)]
    n_proc = len(res)
    start = time.time()

    procs = [mp.Process(target=mp_job, args=(res[i], q, DEFAULT_PADDING, 
        scaling_factor)) for i in range(n_proc)]
    for p in procs:
        p.start()

    logger.info("Starting text recognition in %d thread(s)"%n_proc)
    for p in procs:
        p.join()

    while not q.empty():
        append_result(q.get(), data)
    logger.info("Processed %d frames in %.2f s"%(n_jobs, time.time() - start))

    df = pd.DataFrame(data=data)
    df = df.groupby("timestamp").first().sort_values(by=["timestamp"]).reset_index()
    df.to_csv(output_csv)   

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True, type=str, help="Path to video file")
ap.add_argument("-n", "--n-proc", type=int, default=4, help="Number of cores for multiprocessing")
ap.add_argument("-e", "--skip-extracting", action='store_true', help="Skip extracting images")
ap.add_argument("-p", "--skip-parsing", action='store_true', help="Skip parsing images")
ap.add_argument("-s", "--scaling-factors", type=str, default="5,4,3,2", help="Scaling factor to resize for Tesseract")
ap.add_argument("-d", "--debug-level", type=str, default="info", help="Debug level")
ap.add_argument("-f", "--debug-file", type=str, default="", help="Output logs to file")
ap.add_argument("-o", "--silent", action='store_true', help="Do not output logs to STDOUT")
args = vars(ap.parse_args())

debug_level = logging.WARNING
new_debug_level = "warning" if args["debug_level"] is None else str(args["debug_level"]).lower()
if new_debug_level.startswith("deb"):
    debug_level = logging.DEBUG
elif new_debug_level.startswith("inf"):
    debug_level = logging.INFO
elif new_debug_level.startswith("warn"):
    debug_level = logging.WARNING
elif new_debug_level.startswith("err"):
    debug_level = logging.ERROR
elif new_debug_level.startswith("crit"):
    debug_level = logging.CRITICAL

handlers = []
if not args["silent"]:
    handlers.append(logging.StreamHandler())
if args["debug_file"] and len(args["debug_file"]) > 0:
    if not os.path.exists(args["debug_file"]):
        os.makedirs(args["debug_file"])
    handlers.append(logging.FileHandler(args["debug_file"]))
    
logging.basicConfig(
    level=debug_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=handlers)

logger = logging.getLogger('')
if not os.path.exists(args["input"]) or not os.path.isfile(args["input"]):
    logger.critical("File %s not found, exiting"%args["input"])
    sys.exit()

scaling_factors = list(map(int, args["scaling_factors"].split(",")))
n_proc = args["n_proc"]
output_folder = os.path.splitext(args["input"])[0]
output_csv = output_folder + ".csv"

if not args["skip_extracting"]:
    extract_frames(args["input"], output_folder, n_proc)

if not args["skip_parsing"]:
    parse_folder(output_folder, output_csv, n_proc, scaling_factors)