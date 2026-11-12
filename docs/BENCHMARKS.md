# 引擎对比基线

基线开始：2026-11-12T07:07:27.641641+08:00；最近更新：2026-11-12T07:07:27.641641+08:00。机器同时有其他任务；数字用于留档，空闲时运行 `make bench-compare && make bench-docs` 会覆盖结果及本文件。

所有输入使用固定种子，约 100 个目标持仓；成交价等于收盘价；月度间隔为 21。任务一目标暴露 85%（批量 85%–95%），只收比例佣金 0.025%；任务二初始资金 10 亿（其他任务 1000 万），目标暴露 100%，100 股整手及现金缩减；任务三包含全部可选 A 股面板。批量 K=1000 使用 T=1000、N=100、月度调仓。

运行次数通常为 3，取中位数；超过 60 秒可只测一次。每次独立调用或批量样本设 15 分钟上限（阶段心跳重置计时），超时终止整个进程组，停止该实现更大规模。输入准备及首次调用另列；Numba 首次调用（含可能的编译）不计稳态中位数。无 JIT 的单次首次有效测量若已超过 60 秒，可直接作为唯一样本，并有 sample_includes_first_call 标志。峰值内存包含输入、预热及批量方法试测，不等同胜出方法单独运行的稳态内存；批量子进程内存单列最大单进程值，不冒充聚合值。

|任务|T×N / 调仓|实现|K|回测本身 ms|含输入准备 ms|首次调用 ms|最大净值差|主进程峰值 MiB|最大子进程峰值 MiB|线程/进程|状态|
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
|1|1000×100 / 21|panelbt|1|0.561|1.023|0.614|0|49.2|—|1|ok: same daily NAV within 1e-9|
|1|1000×100 / 21|reference|1|134.099|129.842|133.877|0|49.4|—|1|ok: same daily NAV within 1e-9|
|1|1000×100 / 21|numpy_full|1|0.799|1.012|1.064|2.66e-15|48.7|—|1|ok: same daily NAV within 1e-9|
|1|1000×100 / 21|numpy_sparse|1|1.736|1.779|23.883|2.66e-15|56.5|—|1|ok: same daily NAV within 1e-9|
|1|1000×100 / 21|numpy_approx|1|0.316|0.445|0.702|4.67e-07|53.9|—|1|ok: post-cost allocation / frictionless drift fee approximation; zero-fee control matches|
|1|1000×100 / 21|polars_approx|1|0.914|1.313|62.708|4.67e-07|101.1|—|1|ok: post-cost allocation / frictionless drift fee approximation; zero-fee control matches|
|1|1000×100 / 21|backtrader_strategy|1|1654.905|1399.375|1423.280|2.89e-15|256.8|—|1|ok: same daily NAV within 1e-9|
|1|1000×100 / 21|vectorbt_strategy|1|15.698|15.426|3710.861|1.55e-15|443.2|—|1|ok: same daily NAV within 1e-9|
|1|1000×100 / 21|vectorbt_rust_strategy|1|13.666|14.501|237.462|1.55e-15|338.2|—|1|ok: same daily NAV within 1e-9|
|1|1000×100 / 21|bt_strategy|1|342.750|326.963|329.739|5.02e-07|263.8|—|1|ok: post-cost allocation / frictionless drift fee approximation; zero-fee control matches|
|1|1000×500 / 21|panelbt|1|0.684|1.426|0.846|0|58.8|—|1|ok: same daily NAV within 1e-9|
|1|1000×500 / 21|reference|1|465.959|467.309|467.852|0|58.0|—|1|ok: same daily NAV within 1e-9|
|1|1000×500 / 21|numpy_full|1|1.156|1.211|1.139|3.55e-15|58.0|—|1|ok: same daily NAV within 1e-9|
|1|1000×500 / 21|numpy_sparse|1|2.042|2.125|23.888|3.11e-15|64.7|—|1|ok: same daily NAV within 1e-9|
|1|1000×500 / 21|numpy_approx|1|1.397|1.674|4.319|2.59e-06|76.9|—|1|ok: post-cost allocation / frictionless drift fee approximation; zero-fee control matches|
|1|1000×500 / 21|polars_approx|1|3.160|4.003|90.636|2.59e-06|131.2|—|1|ok: post-cost allocation / frictionless drift fee approximation; zero-fee control matches|
|1|1000×500 / 21|backtrader_strategy|1|6887.106|6950.131|6991.785|5.77e-15|341.6|—|1|ok: same daily NAV within 1e-9|
|1|1000×500 / 21|vectorbt_strategy|1|26.686|26.821|2298.924|5.55e-15|590.6|—|1|ok: same daily NAV within 1e-9|
|1|1000×500 / 21|vectorbt_rust_strategy|1|39.061|37.839|266.054|3.11e-15|451.5|—|1|ok: same daily NAV within 1e-9|
|1|1000×500 / 21|bt_strategy|1|469.125|512.659|462.625|2.82e-06|391.3|—|1|ok: post-cost allocation / frictionless drift fee approximation; zero-fee control matches|
|1|1000×1000 / 21|panelbt|1|0.614|1.729|0.853|0|68.0|—|1|ok: same daily NAV within 1e-9|
|1|1000×1000 / 21|reference|1|852.247|848.873|909.247|0|71.0|—|1|ok: same daily NAV within 1e-9|
|1|1000×1000 / 21|numpy_full|1|1.946|1.903|1.655|5.33e-15|67.0|—|1|ok: same daily NAV within 1e-9|
|1|1000×1000 / 21|numpy_sparse|1|2.869|2.850|26.301|4.88e-15|74.4|—|1|ok: same daily NAV within 1e-9|
|1|1000×1000 / 21|numpy_approx|1|3.196|4.489|8.513|1e-05|105.6|—|1|ok: post-cost allocation / frictionless drift fee approximation; zero-fee control matches|
|1|1000×1000 / 21|polars_approx|1|9.105|9.937|76.722|1e-05|166.5|—|1|ok: post-cost allocation / frictionless drift fee approximation; zero-fee control matches|
|1|1000×1000 / 21|backtrader_strategy|1|13466.926|16256.329|15273.172|4.44e-15|484.0|—|1|ok: same daily NAV within 1e-9|
|1|1000×1000 / 21|vectorbt_strategy|1|47.593|40.533|2425.579|6.22e-15|768.4|—|1|ok: same daily NAV within 1e-9|
|1|1000×1000 / 21|vectorbt_rust_strategy|1|56.993|58.788|284.557|4.44e-15|601.5|—|1|ok: same daily NAV within 1e-9|
|1|1000×1000 / 21|bt_strategy|1|592.535|680.361|610.710|4.08e-06|496.4|—|1|ok: post-cost allocation / frictionless drift fee approximation; zero-fee control matches|
|1|4000×5500 / 21|panelbt|1|2.698|20.456|3.044|0|472.3|—|1|ok: same daily NAV within 1e-9|
|1|4000×5500 / 21|reference|1|17266.653|16579.414|16400.278|0|471.4|—|1|ok: same daily NAV within 1e-9|
|1|4000×5500 / 21|numpy_full|1|14.496|13.839|14.599|1.27e-14|469.8|—|1|ok: same daily NAV within 1e-9|
|1|4000×5500 / 21|numpy_sparse|1|8.990|8.077|29.895|1.15e-14|477.2|—|1|ok: same daily NAV within 1e-9|
|1|4000×5500 / 21|numpy_approx|1|59.904|68.238|168.280|1.06e-05|1309.2|—|1|ok: post-cost allocation / frictionless drift fee approximation; zero-fee control matches|
|1|4000×5500 / 21|polars_approx|1|258.438|273.986|400.700|1.06e-05|1529.9|—|1|ok: post-cost allocation / frictionless drift fee approximation; zero-fee control matches|
|1|4000×5500 / 21|backtrader_strategy|1|330844.682|329111.142|329111.142|1.93e-14|2992.0|—|1|ok: same daily NAV within 1e-9|
|1|4000×5500 / 21|vectorbt_strategy|1|1195.775|1158.822|3477.062|1.18e-14|8368.6|—|1|ok: same daily NAV within 1e-9|
|1|4000×5500 / 21|vectorbt_rust_strategy|1|1286.401|1404.309|1708.561|1.18e-14|6782.5|—|1|ok: same daily NAV within 1e-9|
|1|4000×5500 / 21|bt_strategy|1|7435.173|7417.019|7374.749|5.12e-06|2631.8|—|1|ok: post-cost allocation / frictionless drift fee approximation; zero-fee control matches|
|1|4000×5500 / 1|panelbt|1|27.597|76.039|26.634|0|493.6|—|1|ok: same daily NAV within 1e-9|
|1|4000×5500 / 1|reference|1|21022.775|20153.108|20022.381|0|491.9|—|1|ok: same daily NAV within 1e-9|
|1|4000×5500 / 1|numpy_full|1|191.144|189.283|190.868|3.41e-14|491.8|—|1|ok: same daily NAV within 1e-9|
|1|4000×5500 / 1|numpy_sparse|1|80.747|79.581|105.863|3.25e-14|498.8|—|1|ok: same daily NAV within 1e-9|
|1|4000×5500 / 1|numpy_approx|1|71.038|85.880|158.475|2.34e-05|1331.1|—|1|ok: post-cost allocation / frictionless drift fee approximation; zero-fee control matches|
|1|4000×5500 / 1|polars_approx|1|241.872|261.131|457.721|2.34e-05|1543.8|—|1|ok: post-cost allocation / frictionless drift fee approximation; zero-fee control matches|
|1|4000×5500 / 1|backtrader_strategy|1|335620.118|334459.297|334459.297|6.45e-14|4510.2|—|1|ok: same daily NAV within 1e-9|
|1|4000×5500 / 1|vectorbt_strategy|1|2821.585|2353.079|4119.884|3.26e-14|5935.3|—|1|ok: same daily NAV within 1e-9|
|1|4000×5500 / 1|vectorbt_rust_strategy|1|1700.786|1681.431|1855.326|2.91e-14|6470.2|—|1|ok: same daily NAV within 1e-9|
|1|4000×5500 / 1|bt_strategy|1|44812.205|53265.129|45834.825|4.1e-05|3353.2|—|1|ok: post-cost allocation / frictionless drift fee approximation; zero-fee control matches|
|2|1000×100 / 21|panelbt|1|0.469|0.871|0.554|0|48.9|—|1|ok: same daily NAV within 1e-9|
|2|1000×100 / 21|reference|1|121.651|124.600|128.156|0|48.9|—|1|ok: same daily NAV within 1e-9|
|2|1000×100 / 21|numpy_full|1|0.744|0.744|0.809|1.55e-15|48.7|—|1|ok: same daily NAV within 1e-9|
|2|1000×100 / 21|numpy_sparse|1|1.565|1.626|20.309|1.55e-15|55.5|—|1|ok: same daily NAV within 1e-9|
|2|1000×100 / 21|numpy_approx|1|—|—|—|—|—|—|—|not_applicable: 不支持原生整篮同比例现金缩减与100股申报单位|
|2|1000×100 / 21|polars_approx|1|—|—|—|—|—|—|—|not_applicable: 不支持原生整篮同比例现金缩减与100股申报单位|
|2|1000×100 / 21|backtrader_strategy|1|1145.284|1142.402|1166.109|2e-15|254.8|—|1|ok: same daily NAV within 1e-9|
|2|1000×100 / 21|vectorbt_strategy|1|10.520|11.018|1956.890|4.44e-16|438.0|—|1|ok: same daily NAV within 1e-9|
|2|1000×100 / 21|vectorbt_rust_strategy|1|—|—|—|—|—|—|—|not_applicable: 不支持原生整篮同比例现金缩减与100股申报单位|
|2|1000×100 / 21|bt_strategy|1|168.917|198.567|210.255|4.44e-16|262.5|—|1|ok: same daily NAV within 1e-9|
|2|1000×500 / 21|panelbt|1|0.570|1.298|0.733|0|57.7|—|1|ok: same daily NAV within 1e-9|
|2|1000×500 / 21|reference|1|424.161|434.438|427.602|0|58.3|—|1|ok: same daily NAV within 1e-9|
|2|1000×500 / 21|numpy_full|1|1.074|1.106|1.163|3.89e-15|56.4|—|1|ok: same daily NAV within 1e-9|
|2|1000×500 / 21|numpy_sparse|1|1.952|1.991|21.434|4.55e-15|63.9|—|1|ok: same daily NAV within 1e-9|
|2|1000×500 / 21|numpy_approx|1|—|—|—|—|—|—|—|not_applicable: 不支持原生整篮同比例现金缩减与100股申报单位|
|2|1000×500 / 21|polars_approx|1|—|—|—|—|—|—|—|not_applicable: 不支持原生整篮同比例现金缩减与100股申报单位|
|2|1000×500 / 21|backtrader_strategy|1|5532.772|5748.683|5412.172|2.66e-15|332.2|—|1|ok: same daily NAV within 1e-9|
|2|1000×500 / 21|vectorbt_strategy|1|19.351|20.052|1832.836|4.44e-16|582.7|—|1|ok: same daily NAV within 1e-9|
|2|1000×500 / 21|vectorbt_rust_strategy|1|—|—|—|—|—|—|—|not_applicable: 不支持原生整篮同比例现金缩减与100股申报单位|
|2|1000×500 / 21|bt_strategy|1|250.061|247.482|252.018|4.44e-16|392.1|—|1|ok: same daily NAV within 1e-9|
|2|1000×1000 / 21|panelbt|1|0.535|1.484|0.683|0|67.0|—|1|ok: same daily NAV within 1e-9|
|2|1000×1000 / 21|reference|1|673.297|684.790|677.950|0|67.1|—|1|ok: same daily NAV within 1e-9|
|2|1000×1000 / 21|numpy_full|1|1.642|1.669|1.689|1.78e-15|66.6|—|1|ok: same daily NAV within 1e-9|
|2|1000×1000 / 21|numpy_sparse|1|1.784|1.873|19.718|2.22e-15|73.0|—|1|ok: same daily NAV within 1e-9|
|2|1000×1000 / 21|numpy_approx|1|—|—|—|—|—|—|—|not_applicable: 不支持原生整篮同比例现金缩减与100股申报单位|
|2|1000×1000 / 21|polars_approx|1|—|—|—|—|—|—|—|not_applicable: 不支持原生整篮同比例现金缩减与100股申报单位|
|2|1000×1000 / 21|backtrader_strategy|1|11427.713|10990.789|9784.335|2.89e-15|466.2|—|1|ok: same daily NAV within 1e-9|
|2|1000×1000 / 21|vectorbt_strategy|1|33.263|33.452|1983.749|8.88e-16|766.0|—|1|ok: same daily NAV within 1e-9|
|2|1000×1000 / 21|vectorbt_rust_strategy|1|—|—|—|—|—|—|—|not_applicable: 不支持原生整篮同比例现金缩减与100股申报单位|
|2|1000×1000 / 21|bt_strategy|1|356.229|349.572|342.156|4.44e-16|463.1|—|1|ok: same daily NAV within 1e-9|
|2|4000×5500 / 21|panelbt|1|2.526|17.872|2.656|0|471.1|—|1|ok: same daily NAV within 1e-9|
|2|4000×5500 / 21|reference|1|14891.669|15006.434|15403.848|0|469.8|—|1|ok: same daily NAV within 1e-9|
|2|4000×5500 / 21|numpy_full|1|14.241|14.071|14.438|6.88e-15|468.5|—|1|ok: same daily NAV within 1e-9|
|2|4000×5500 / 21|numpy_sparse|1|8.009|8.026|33.431|5.11e-15|475.9|—|1|ok: same daily NAV within 1e-9|
|2|4000×5500 / 21|numpy_approx|1|—|—|—|—|—|—|—|not_applicable: 不支持原生整篮同比例现金缩减与100股申报单位|
|2|4000×5500 / 21|polars_approx|1|—|—|—|—|—|—|—|not_applicable: 不支持原生整篮同比例现金缩减与100股申报单位|
|2|4000×5500 / 21|backtrader_strategy|1|278463.508|293375.708|293375.708|6e-15|3430.7|—|1|ok: same daily NAV within 1e-9|
|2|4000×5500 / 21|vectorbt_strategy|1|993.902|937.173|3113.465|8.88e-16|6496.1|—|1|ok: same daily NAV within 1e-9|
|2|4000×5500 / 21|vectorbt_rust_strategy|1|—|—|—|—|—|—|—|not_applicable: 不支持原生整篮同比例现金缩减与100股申报单位|
|2|4000×5500 / 21|bt_strategy|1|3636.094|3927.603|3868.337|4.44e-16|2640.7|—|1|ok: same daily NAV within 1e-9|
|2|4000×5500 / 1|panelbt|1|24.179|64.013|24.521|0|493.9|—|1|ok: same daily NAV within 1e-9|
|2|4000×5500 / 1|reference|1|17302.503|18310.983|17035.355|0|492.5|—|1|ok: same daily NAV within 1e-9|
|2|4000×5500 / 1|numpy_full|1|196.936|196.567|196.050|6.14e-14|490.2|—|1|ok: same daily NAV within 1e-9|
|2|4000×5500 / 1|numpy_sparse|1|69.233|69.291|89.120|6.02e-14|497.1|—|1|ok: same daily NAV within 1e-9|
|2|4000×5500 / 1|numpy_approx|1|—|—|—|—|—|—|—|not_applicable: 不支持原生整篮同比例现金缩减与100股申报单位|
|2|4000×5500 / 1|polars_approx|1|—|—|—|—|—|—|—|not_applicable: 不支持原生整篮同比例现金缩减与100股申报单位|
|2|4000×5500 / 1|backtrader_strategy|1|272014.909|260715.439|260715.439|7.11e-15|7075.6|—|1|ok: same daily NAV within 1e-9|
|2|4000×5500 / 1|vectorbt_strategy|1|2232.288|2025.091|3798.568|1.89e-15|7903.8|—|1|ok: same daily NAV within 1e-9|
|2|4000×5500 / 1|vectorbt_rust_strategy|1|—|—|—|—|—|—|—|not_applicable: 不支持原生整篮同比例现金缩减与100股申报单位|
|2|4000×5500 / 1|bt_strategy|1|12960.473|13790.162|14216.022|3.33e-16|3577.3|—|1|ok: same daily NAV within 1e-9|
|3|1000×100 / 21|panelbt|1|0.498|1.044|0.581|0|52.2|—|1|ok: same daily NAV within 1e-9|
|3|1000×100 / 21|reference|1|144.223|149.587|148.452|0|52.3|—|1|ok: same daily NAV within 1e-9|
|3|1000×100 / 21|numpy_full|1|—|—|—|—|—|—|—|not_applicable: 完整规则仅测 panelbt 和 reference|
|3|1000×100 / 21|numpy_sparse|1|—|—|—|—|—|—|—|not_applicable: 完整规则仅测 panelbt 和 reference|
|3|1000×100 / 21|numpy_approx|1|—|—|—|—|—|—|—|not_applicable: 完整规则仅测 panelbt 和 reference|
|3|1000×100 / 21|polars_approx|1|—|—|—|—|—|—|—|not_applicable: 完整规则仅测 panelbt 和 reference|
|3|1000×100 / 21|backtrader_strategy|1|—|—|—|—|—|—|—|not_applicable: 完整规则仅测 panelbt 和 reference|
|3|1000×100 / 21|vectorbt_strategy|1|—|—|—|—|—|—|—|not_applicable: 完整规则仅测 panelbt 和 reference|
|3|1000×100 / 21|vectorbt_rust_strategy|1|—|—|—|—|—|—|—|not_applicable: 完整规则仅测 panelbt 和 reference|
|3|1000×100 / 21|bt_strategy|1|—|—|—|—|—|—|—|not_applicable: 完整规则仅测 panelbt 和 reference|
|3|1000×500 / 21|panelbt|1|0.630|2.149|0.719|0|73.6|—|1|ok: same daily NAV within 1e-9|
|3|1000×500 / 21|reference|1|448.299|455.116|457.058|0|73.2|—|1|ok: same daily NAV within 1e-9|
|3|1000×500 / 21|numpy_full|1|—|—|—|—|—|—|—|not_applicable: 完整规则仅测 panelbt 和 reference|
|3|1000×500 / 21|numpy_sparse|1|—|—|—|—|—|—|—|not_applicable: 完整规则仅测 panelbt 和 reference|
|3|1000×500 / 21|numpy_approx|1|—|—|—|—|—|—|—|not_applicable: 完整规则仅测 panelbt 和 reference|
|3|1000×500 / 21|polars_approx|1|—|—|—|—|—|—|—|not_applicable: 完整规则仅测 panelbt 和 reference|
|3|1000×500 / 21|backtrader_strategy|1|—|—|—|—|—|—|—|not_applicable: 完整规则仅测 panelbt 和 reference|
|3|1000×500 / 21|vectorbt_strategy|1|—|—|—|—|—|—|—|not_applicable: 完整规则仅测 panelbt 和 reference|
|3|1000×500 / 21|vectorbt_rust_strategy|1|—|—|—|—|—|—|—|not_applicable: 完整规则仅测 panelbt 和 reference|
|3|1000×500 / 21|bt_strategy|1|—|—|—|—|—|—|—|not_applicable: 完整规则仅测 panelbt 和 reference|
|3|1000×1000 / 21|panelbt|1|0.575|3.052|0.716|0|98.7|—|1|ok: same daily NAV within 1e-9|
|3|1000×1000 / 21|reference|1|827.173|834.343|857.778|0|98.7|—|1|ok: same daily NAV within 1e-9|
|3|1000×1000 / 21|numpy_full|1|—|—|—|—|—|—|—|not_applicable: 完整规则仅测 panelbt 和 reference|
|3|1000×1000 / 21|numpy_sparse|1|—|—|—|—|—|—|—|not_applicable: 完整规则仅测 panelbt 和 reference|
|3|1000×1000 / 21|numpy_approx|1|—|—|—|—|—|—|—|not_applicable: 完整规则仅测 panelbt 和 reference|
|3|1000×1000 / 21|polars_approx|1|—|—|—|—|—|—|—|not_applicable: 完整规则仅测 panelbt 和 reference|
|3|1000×1000 / 21|backtrader_strategy|1|—|—|—|—|—|—|—|not_applicable: 完整规则仅测 panelbt 和 reference|
|3|1000×1000 / 21|vectorbt_strategy|1|—|—|—|—|—|—|—|not_applicable: 完整规则仅测 panelbt 和 reference|
|3|1000×1000 / 21|vectorbt_rust_strategy|1|—|—|—|—|—|—|—|not_applicable: 完整规则仅测 panelbt 和 reference|
|3|1000×1000 / 21|bt_strategy|1|—|—|—|—|—|—|—|not_applicable: 完整规则仅测 panelbt 和 reference|
|3|4000×5500 / 21|panelbt|1|3.216|57.992|3.647|0|1164.0|—|1|ok: same daily NAV within 1e-9|
|3|4000×5500 / 21|reference|1|17542.727|17098.585|17542.089|0|1162.8|—|1|ok: same daily NAV within 1e-9|
|3|4000×5500 / 21|numpy_full|1|—|—|—|—|—|—|—|not_applicable: 完整规则仅测 panelbt 和 reference|
|3|4000×5500 / 21|numpy_sparse|1|—|—|—|—|—|—|—|not_applicable: 完整规则仅测 panelbt 和 reference|
|3|4000×5500 / 21|numpy_approx|1|—|—|—|—|—|—|—|not_applicable: 完整规则仅测 panelbt 和 reference|
|3|4000×5500 / 21|polars_approx|1|—|—|—|—|—|—|—|not_applicable: 完整规则仅测 panelbt 和 reference|
|3|4000×5500 / 21|backtrader_strategy|1|—|—|—|—|—|—|—|not_applicable: 完整规则仅测 panelbt 和 reference|
|3|4000×5500 / 21|vectorbt_strategy|1|—|—|—|—|—|—|—|not_applicable: 完整规则仅测 panelbt 和 reference|
|3|4000×5500 / 21|vectorbt_rust_strategy|1|—|—|—|—|—|—|—|not_applicable: 完整规则仅测 panelbt 和 reference|
|3|4000×5500 / 21|bt_strategy|1|—|—|—|—|—|—|—|not_applicable: 完整规则仅测 panelbt 和 reference|
|3|4000×5500 / 1|panelbt|1|19.941|99.709|21.921|0|1186.2|—|1|ok: same daily NAV within 1e-9|
|3|4000×5500 / 1|reference|1|25242.249|23750.421|19936.139|0|1184.7|—|1|ok: same daily NAV within 1e-9|
|3|4000×5500 / 1|numpy_full|1|—|—|—|—|—|—|—|not_applicable: 完整规则仅测 panelbt 和 reference|
|3|4000×5500 / 1|numpy_sparse|1|—|—|—|—|—|—|—|not_applicable: 完整规则仅测 panelbt 和 reference|
|3|4000×5500 / 1|numpy_approx|1|—|—|—|—|—|—|—|not_applicable: 完整规则仅测 panelbt 和 reference|
|3|4000×5500 / 1|polars_approx|1|—|—|—|—|—|—|—|not_applicable: 完整规则仅测 panelbt 和 reference|
|3|4000×5500 / 1|backtrader_strategy|1|—|—|—|—|—|—|—|not_applicable: 完整规则仅测 panelbt 和 reference|
|3|4000×5500 / 1|vectorbt_strategy|1|—|—|—|—|—|—|—|not_applicable: 完整规则仅测 panelbt 和 reference|
|3|4000×5500 / 1|vectorbt_rust_strategy|1|—|—|—|—|—|—|—|not_applicable: 完整规则仅测 panelbt 和 reference|
|3|4000×5500 / 1|bt_strategy|1|—|—|—|—|—|—|—|not_applicable: 完整规则仅测 panelbt 和 reference|
|1|1000×100 / 21|panelbt|1000|56.287|341.144|0.558|0|814.9|—|15|ok: same daily NAV within 1e-9|
|1|1000×100 / 21|reference|1000|13159.741|13635.214|122.250|0|201.2|139.7|15|ok: same daily NAV within 1e-9|
|1|1000×100 / 21|numpy_full|1000|92.603|83.979|0.741|8.84e-14|221.8|142.5|15|ok: same daily NAV within 1e-9|
|1|1000×100 / 21|numpy_sparse|1000|204.241|174.268|19.751|8.84e-14|223.6|152.2|15|ok: same daily NAV within 1e-9|
|1|1000×100 / 21|numpy_approx|1000|52.973|62.965|0.661|5.83e-07|238.4|164.8|15|ok: post-cost allocation / frictionless drift fee approximation; zero-fee control matches|
|1|1000×100 / 21|polars_approx|1000|572.250|682.949|48.548|5.83e-07|3374.9|204.2|1|ok: post-cost allocation / frictionless drift fee approximation; zero-fee control matches|
|1|1000×100 / 21|backtrader_strategy|1000|139616.638|130672.329|1104.620|8.79e-14|374.3|502.7|15|ok: same daily NAV within 1e-9|
|1|1000×100 / 21|vectorbt_strategy|1000|1077.757|1140.221|1777.484|8.86e-14|824.5|657.7|15|ok: same daily NAV within 1e-9|
|1|1000×100 / 21|vectorbt_rust_strategy|1000|1272.450|1164.354|166.612|8.93e-14|728.1|605.9|15|ok: same daily NAV within 1e-9|
|1|1000×100 / 21|bt_strategy|1000|32264.598|30149.977|270.083|5.61e-07|462.5|432.5|15|ok: post-cost allocation / frictionless drift fee approximation; zero-fee control matches|

## 语义与公平性

- panelbt、reference、NumPy 逐日两版均核对净值，阈值 rtol=atol=1e-9。任务一每个第三方实现也核对逐日净值。
- NumPy/Polars 整块近似版在调仓间隔中保留价格漂移，用无摩擦权重估计换手，扣费后按比例缩放财富；panelbt 按扣费前权益确定股数。bt 在各子资产分配额内扣佣金，同样有扣费口径差异。三者另跑零费率控制，必须逐日一致才允许报告耗时；实际差额列在表中。
- vectorbt Numba 使用只在调仓日触发的 pre_segment/order 回调；其批量和可选 Rust 引擎使用原生 targetpercent、共享现金、退出标的优先的显式调用顺序和固定当行估值（任务一有 5%–15% 现金缓冲，其余目标持仓内调整不会受现金约束；仍逐日核对净值）；backtrader 使用 preload/runonce、cheat-on-open、无 observers、种子 bar 对齐首日，不把 Python 自写简化回测冒充第三方内核。
- 任务二：backtrader 用 next_open 进行篮子预算，bt 用 Algo/原生 transact，vectorbt 用 Numba pre_segment/order 回调实现整手及同比例缩减，成交和账户记账仍由各自引擎完成。vectorbt Rust 的自定义 Numba 回调路径不适用；NumPy/Polars 整块近似版不能表达跨日整手与现金状态。
- 完整规则中，前日持仓、现金预算、取整、阻塞顺延和公司行为共同决定下一日状态；当前 Polars 向量表达式没有这种组合状态的递推接口。用 Python UDF 包装逐日循环不作为 Polars 向量化实现参与比较。
- 近似模式的批量也逐一核对全部 1000 组零费率曲线，并检查两种计时口径的含费结果一致；净值差列取全部组合的最大值。1000 组批量使用各引擎的原生批量（Rayon、vectorbt 分组）或串行/进程试测后的胜者；每组目标内容哈希均不同（N=H=100 时以 1000 档暴露区分）。进程预热不计稳态；批量数据在各工作进程初始化一次。
- 两种耗时均直接重复测量：回测本身使用预先转换的不可变输入；含输入准备从统一 NumPy 行情与 CSR 目标出发，panelbt/reference 计入 Market/Targets 校验，对手计入自己的 DataFrame/稠密矩阵转换。随机数据生成、模块导入和 JIT 预热不计；引擎对象实例化和结果提取计入。批量行情每个工作进程初始化一次，含输入准备口径计逐计划转换；预备输入的独立工作进程固定分配目标，避免缓存遗漏。reference 的独立费用展开仍保留在执行内。
- 两种口径在不同时间窗口取样；忙机上的中位数可能倒置，不能相减后把负差解释成准备耗时。原始样本完整保留，空闲机器重跑后再作细小差异判断。
- 版本、原始样本、首次编译/调用、内存、批量试测、失败日志均在 `bench/results/compare.json` 及 `bench/results/logs/`。

## 复现

`make dev && make bench-compare`；沙箱可用 `make dev SANDBOX=1`。`make bench` 只测自家引擎，并与仓库基线比较，退化超过 10% 标红字样 REGRESSION。

适配器依据：[Backtrader Cheat-On-Open](https://backtrader.com/blog/posts/2017-05-01-cheat-on-open/cheat-on-open/)、[vectorbt Portfolio](https://vectorbt.dev/api/portfolio/base/)、[bt API](https://pmorissette.github.io/bt/bt.html)。

机器与版本：
```json
{
  "platform": "macOS-26.6-arm64-arm-64bit-Mach-O",
  "processor": "arm",
  "hardware_details": "sysctl restricted by sandbox; architecture and CPU count remain available",
  "python": "3.13.13",
  "cpus": 15,
  "versions": {
    "numpy": "2.5.4",
    "panelbt": "0.1.0",
    "polars": "2.1.0",
    "pandas": "3.1.1",
    "backtrader": "1.9.78.123",
    "vectorbt": "1.2.1",
    "vectorbt-rust": "1.2.1",
    "bt": "1.4.1",
    "numba": "0.68.1"
  },
  "concurrent_workload": true
}
```

## 自家引擎与回归检查

|场景|中位数 ms|相对仓库基线|
|---|---:|---:|
|lots|2.101|-50.6%|
|fractional|1.871|-43.7%|
|batch|472.395|-12.6%|
|long_short|3.579|-0.4%|

## 步骤 1c：期货对冲腿

4000×5500 现货面板加 20 列期货、约 100 只现货、每 21 日调仓；同一已校验市场对象、9 次交替调用、单线程。完整规则的 NumPy 对照不适用：下方 NumPy 基线只实现同义的简化对冲，不含保证金、到期、强平或公司行为。市场构建及输入转换不计入时间。原始采样写入被忽略的 `.cache/panelbt/futures-current.json`；当前机器有并行任务，空闲时用 `make bench && make bench-docs` 重测。

| 任务 | panelbt 最小 ms | panelbt 中位数 ms | NumPy 中位数 ms | 净值最大误差 |
|---|---:|---:|---:|---:|
| 完整现货规则 | 1.774 | 2.081 | 不适用 | — |
| 完整现货+20 列期货 | 3.077 | 3.528 | 不适用 | — |
| 同义简化对冲任务 | 0.982 | 1.015 | 0.562 | 3.33e-15 |

简化任务使用活跃证券稀疏 NumPy 运算；该任务 NumPy 比自研快，不能用它外推完整规则。新增混合语料 20000 组、227243 笔成交：参考实现 0 分叉，独立账本 0 不符。复现：`uv run --no-sync python bench/validate.py --corpus futures --start 40000 --count 20000`。

机器：macOS-26.6-arm64-arm-64bit-Mach-O，Python 3.13.13，NumPy 2.5.4；开始时 1/5/15 分钟负载 2.41, 2.41, 2.60，峰值 RSS 2.07 GiB。

同输入现货路径与 1b `4d080d0` 各自 LTO 构建、AB/BA 交替配对（4000×5500，不含准备）；三场景最小值比均小于 1.05：

| 场景 | 1b 最小 / 中位 ms | 当前最小 / 中位 ms | 最小值比 |
|---|---:|---:|---:|
| 完整规则含退市 | 2.247 / 2.631 | 1.713 / 1.774 | 0.762 |
| 不传退市面板 | 2.547 / 2.662 | 1.715 / 1.817 | 0.673 |
| 仅价格 | 1.957 / 2.023 | 1.662 / 1.715 | 0.849 |
