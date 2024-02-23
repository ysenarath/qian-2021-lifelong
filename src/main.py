import argparse
from pathlib import Path

from src.vae_lbsoinn import run_sequence


def main():
    parser = argparse.ArgumentParser(description="-----[rnn-classifier]-----")
    parser.add_argument(
        "--model",
        default="rand",
        help="available models: rand, static, non-static, multichannel",
    )
    parser.add_argument(
        "--vae_model", default="mask", help="available models: original, mask"
    )
    parser.add_argument(
        "--save_model",
        default=False,
        action="store_true",
        help="whether saving model or not",
    )
    parser.add_argument("--epoch", default=20, type=int, help="number of max epoch")
    parser.add_argument(
        "--learning_rate", default=1e-4, type=float, help="learning rate"
    )
    parser.add_argument(
        "--hidden_size",
        default=64,
        type=int,
        help="size of hidden layer of lstm",
    )
    parser.add_argument("--num_layers", default=1, type=int, help="num of lstm layers")
    parser.add_argument(
        "--gpu", default=0, type=int, help="the number of gpu to be used"
    )
    parser.add_argument(
        "--word_dim",
        default=300,
        type=int,
        help="dimension of word embeddings",
    )
    parser.add_argument("--batch_size", default=64, type=int, help="batch size")
    parser.add_argument(
        "--max_seq_length",
        default=40,
        type=int,
        help="maximum number of tokens of the input",
    )
    parser.add_argument(
        "--lower_case",
        default=False,
        action="store_true",
        help="lower case input",
    )
    parser.add_argument(
        "--dropout", default=0.3, type=float, help="dropout probability"
    )
    parser.add_argument(
        "--loss_margin",
        default=0.5,
        type=float,
        help="margin ranking loss param",
    )
    parser.add_argument(
        "--training_num_limit",
        default=5000,
        type=int,
        help="limit of training instances per task",
    )
    parser.add_argument(
        "--similarity",
        default="cos",
        type=str,
        help="vector similarity metric: cos or l2",
    )
    parser.add_argument("--memory_size", default=1000, type=int, help="memory size")
    parser.add_argument(
        "--memory_replay",
        default="lbsoinn",
        type=str,
        help="strategy to write mem: lbsoinn or rand",
    )
    parser.add_argument("--fea_loss", default="cos", type=str, help="cos or mse")

    parser.add_argument("--train_path", type=str, help="path of training data")
    parser.add_argument("--test_path", type=str, help="path of testing data")
    parser.add_argument("--dev_path", type=str, help="path of development data")
    parser.add_argument(
        "--balance",
        default=False,
        action="store_true",
        help="do balanced sampling for training data",
    )
    parser.add_argument(
        "--forward",
        default=False,
        action="store_true",
        help="do forward prediction",
    )
    parser.add_argument(
        "--cand_limit",
        type=int,
        help="should be consisitent with training testing dev path",
    )

    parser.add_argument(
        "--iter_thres",
        default=1000,
        type=int,
        help="lbsoinn iteration threshold",
    )
    parser.add_argument(
        "--max_edge_age", default=50, type=int, help="lbsoinn maximum edge age"
    )
    parser.add_argument("--c1", default=0.001, type=float, help="lbsoinn c1")
    parser.add_argument("--c2", default=1.0, type=float, help="lbsoinn c2")
    parser.add_argument("--gamma", default=1.04, type=float, help="lbsoinn gamma")
    parser.add_argument("--lw1", default=20, type=float, help="loss weight 1")

    parser.add_argument(
        "--keep_all_node",
        default=False,
        action="store_true",
        help="disable deleting noisy nodes",
    )
    parser.add_argument(
        "--mix_train",
        default=False,
        action="store_true",
        help="mix the training data with memory data",
    )
    parser.add_argument(
        "--fix_per_task_mem_size",
        default=False,
        action="store_true",
        help="fix the per task memory size",
    )

    options = parser.parse_args()

    training_config = {
        "save_model": options.save_model,
        "epoch": options.epoch,
        "lr": options.learning_rate,
        "max_seq_length": options.max_seq_length,
        "batch_size": options.batch_size,
        "lower_case": options.lower_case,
        "gpu": options.gpu,
        "training_num_limit": options.training_num_limit,
        "balance": options.balance,
        "forward": options.forward,
        "cand_limit": options.cand_limit,
        "memory_size": options.memory_size,
        "lw1": options.lw1,
        "fea_loss": options.fea_loss,
        "mix_train": options.mix_train,
        "fix_per_task_mem_size": options.fix_per_task_mem_size,
        "memory_replay": options.memory_replay,
    }
    model_config = {
        "model": options.model,
        "vae_model": options.vae_model,
        "word_dim": options.word_dim,
        "hidden_size": options.hidden_size,
        "num_layers": options.num_layers,
        "dropout_prob": options.dropout,
        "loss_margin": options.loss_margin,
        "similarity": options.similarity,
        "gpu": options.gpu,
        "iter_thres": options.iter_thres,
        "max_edge_age": options.max_edge_age,
        "c1": options.c1,
        "c2": options.c2,
        "keep_node": options.keep_all_node,
        "gamma": options.gamma,
    }
    cat_order = []
    order_fp = Path(options.train_path).parent / "order.txt"
    with open(order_fp, "r") as f:
        for line in f:
            cat_order.append(line.strip())
    # cat_order = [
    #     "anti_immigration.csv",
    #     "anti_semitism.csv",
    #     "hate_music.csv",
    #     "anti_catholic.csv",
    #     "ku_klux_klan.csv",
    #     "anti_muslim.csv",
    #     "black_separatist.csv",
    #     "white_nationalist.csv",
    #     "neo_nazi.csv",
    #     "anti_lgbtq.csv",
    #     "christian_identity.csv",
    #     "holocaust_identity.csv",
    #     "neo_confederate.csv",
    #     "racist_skinhead.csv",
    #     "radical_traditional_catholic.csv",
    # ]
    # cat_order = ['neo_nazi.csv',
    #              'racist_skinhead.csv',
    #              'anti_semitism.csv',
    #              'ku_klux_klan.csv',
    #              'white_nationalist.csv',
    #              'anti_immigration.csv',
    #              'anti_muslim.csv',
    #              'anti_catholic.csv',
    #              'radical_traditional_catholic.csv',
    #              'anti_lgbtq.csv',
    #              'neo_confederate.csv',
    #              'holocaust_identity.csv',
    #              'hate_music.csv',
    #              'black_separatist.csv',
    #              'christian_identity.csv']

    # cat_order=['ku_klux_klan.csv',
    #            'anti_semitism.csv',
    #            'black_separatist.csv',
    #            'white_nationalist.csv',
    #            'neo_nazi.csv',
    #            'hate_music.csv',
    #            'christian_identity.csv',
    #            'anti_immigration.csv',
    #            'holocaust_identity.csv',
    #            'neo_confederate.csv',
    #            'anti_muslim.csv',
    #            'anti_lgbtq.csv',
    #            'anti_catholic.csv',
    #            'racist_skinhead.csv',
    #            'radical_traditional_catholic.csv']
    # cat_order = ['anti_semitism.csv',
    #              'neo_confederate.csv',
    #              'anti_immigration.csv',
    #              'white_nationalist.csv',
    #              'neo_nazi.csv',
    #              'christian_identity.csv',
    #              'anti_catholic.csv',
    #              'holocaust_identity.csv',
    #              'radical_traditional_catholic.csv',
    #              'anti_lgbtq.csv',
    #              'black_separatist.csv',
    #              'hate_music.csv',
    #              'anti_muslim.csv',
    #              'racist_skinhead.csv',
    #              'ku_klux_klan.csv']
    print("=" * 20 + "INFORMATION" + "=" * 20)
    print(training_config)
    print(model_config)
    print("\t".join(cat_order))
    print(options.train_path)
    print("=" * 20 + "INFORMATION" + "=" * 20)
    print("=" * 20 + "TRAINING STARTED" + "=" * 20)
    model = run_sequence(
        options.train_path,
        options.test_path,
        options.dev_path,
        cat_order,
        training_config,
        model_config,
    )
    print("=" * 20 + "TRAINING FINISHED" + "=" * 20)


if __name__ == "__main__":
    main()
