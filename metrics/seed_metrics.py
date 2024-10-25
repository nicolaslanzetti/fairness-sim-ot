from utils.housekeeping_utils import hk_init

PROJECT_ROOT, global_logger = hk_init()


def calculate_seedset_performance(multi_run_seed_info: list,
                                  logger=global_logger):
    """records outreach summary statistics resulting from a seedset's outreach
    """
    aggr_seed_perf = {}
    seedset = list(multi_run_seed_info.keys()
                   )  # multi_run_seed_info is seed X runs X seed_info_struct
    uniq_groups = set(multi_run_seed_info[seedset[0]][0][1].keys())
    n_times = len(multi_run_seed_info[seedset[0]])

    for seed in seedset:
        aggr_seed_perf[seed] = [
            0,
            dict(
                zip(list(uniq_groups),
                    [[0, 0] for _ in range(len(uniq_groups))]))
        ]

    # TODO(schowdhary): make sure the metrics still make sense
    for seed in seedset:
        tot_seed_reach = 0
        tot_seed_reach_aggr_calc = 0
        for res_run in multi_run_seed_info[seed]:
            tot_seed_reach += res_run[0]
            aggr_seed_perf[seed][0] += res_run[0]

            assert uniq_groups == set(res_run[1].keys())
            for group in uniq_groups:
                tot_seed_reach_aggr_calc += res_run[1][group][0]
                aggr_seed_perf[seed][1][group][0] += res_run[1][group][0]
                aggr_seed_perf[seed][1][group][1] += res_run[1][group][
                    0] * res_run[1][group][1]  # weighted sum

        assert tot_seed_reach == tot_seed_reach_aggr_calc
        assert aggr_seed_perf[seed][0] == tot_seed_reach
        aggr_seed_perf[seed][0] /= n_times

        for group in uniq_groups:
            if aggr_seed_perf[seed][1][group][0]:
                aggr_seed_perf[seed][1][group][1] /= aggr_seed_perf[seed][1][
                    group][0]
            else:
                aggr_seed_perf[seed][1][group][1] = 0
            aggr_seed_perf[seed][1][group][0] /= n_times

    logger.info("==================== <SEED METRICS ====================")
    logger.info("Total realizations: %d", n_times)
    for seed in seedset:
        logger.info("\n")
        logger.info("<For seed %s: ====================", seed)
        logger.info("\n")
        logger.info("Average nodes reach: %f", aggr_seed_perf[seed][0])
        for group in uniq_groups:
            logger.info("\n")
            logger.info("Reaching group index, %d", group)
            logger.info("Average group nodes reached, %f",
                        aggr_seed_perf[seed][1][group][0])
            logger.info("Average time steps for reached group nodes, %f",
                        aggr_seed_perf[seed][1][group][1])
        logger.info("\n")
        logger.info("For seed %s> ====================", seed)
    logger.info("\n")
    logger.info("==================== SEED METRICS> ====================")
    return aggr_seed_perf
