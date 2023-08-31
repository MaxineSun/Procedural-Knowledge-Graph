import utils.parse_args as pa
import utils.pre_sq_unfolded as psu
import utils.rw_score as rs
import utils.sq_ER_GPT as sq
import pickle


def main(args):
    p = psu.Data_Process()
    p.json2dataset(args)


if __name__ == "__main__":
    args = pa.parse_args()
    main(args)
