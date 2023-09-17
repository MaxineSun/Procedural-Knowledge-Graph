import utils.parse_args as pa
import utils.pre_sq_unfolded as psu
import utils.json2heter as j2h
import utils.rw_score as rs
# import utils.sq_ER_GPT as sq
import utils.sq_dbscan as sq
import pickle


def main(args):
    # p = sq.DBScan_Process()
    # p.Clustering(0.8)
    p = psu.Data_Process()
    p.json2dataset(args)


if __name__ == "__main__":
    args = pa.parse_args()
    main(args)
