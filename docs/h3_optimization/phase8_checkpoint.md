# H3 第八阶段已验证检查点

更新时间：2026-11-09T08:12:13.788396+08:00（北京时间）。

本Git分支保存独立优化副本，历史父提交为原始复制时的834488108a0af0636590e2349240ac336163ecc5。正式工作线submit_repo独立维护；本分支不继承后来正式入口重构。已验节点环境来自AFS final-v2父包与增量，并非仅凭该源码目录即可冷启动。

36组注意力局部检查、VAE串行/2/4/8卡，以及三组完整FL2VA生成均通过。原生v4→m128-n32-w4-s2+4卡VAE：稳定步5.095→4.395秒，主流程297.54→238.92秒，外层子进程391.89→346.09秒；视频SSIM96.667%，音频PCM相对L2差异9.953%。仅4卡VAE的同latent输出逐位一致。注意力候选同完整DiT输入的音频输出相对v4差异1.975%，没有完成音质验收。完整配置各一轮，仍有逐步同步计时探针；比较对象为本轮native v4，不是官方PyTorch同轮reference。

已验配置：既有三融合打开，AOT注意力打开，m128-n32-w4-s2，VAE设备数4。候选仍默认关闭，运行时须显式选择。源码14文件SHA与本地v3清单一致。性能、缓存和释放收据保存在phase8/；完整视频/插件及归档仍在实验目录的phase8_full/、phase8_validation4/、phase8_audio/。

必须保留AFS整个h3-cache/h3-opt-phase67-20261108/，尤其final-v2与round2-v4-delta两包；本地phase8_audio/audio8-native-delta-5fd90948.tar.gz及收据、restore脚本缺一不可。1206对象跨新sid恢复零对象编译已验，存在必要重链接。本轮AFS只读，所有资源已释放。

配置plugin SHA=7737367a5370844f62e26892bf856d6ee708b706f538b2e44663a77d22816a0e。
v4 plugin SHA=05d78dddba8d4c8a0a8e90e99cceee19654cf44056e7639aedb76c5a3310eefe。
完整结果归档SHA=edbf001edf85a82cbdaec58faa19e6360712d5697546d8afb64b3033eed4a177。
