# 固定 FL2VA 参考视频

reference_fl2va.mp4 是历史完整官方 legacy 参考端生成的视频，不是本轮生成的 native 视频。
case.json 登记其图片大小、Prompt、seed42、50步请求、124帧、参考视频大小和原始运行标识。

本目录随源码仓库一起上传到个人 AFS。bash run_submission.sh check
会完整生成新的 native 视频，再使用本参考视频调用未修改官方比较器；reference 不会重新生成。
默认不加 check 时，只生成并保存 native 视频。

本视频只适用于 case.json 对应的测试用例。换图片或官方参数后，需要对应的新参考视频与登记信息，
将新视频和登记信息更新到AFS源码的reference目录；不能拿本视频比较其他用例，也不能以复用参考视频的结果宣称同轮双方加速比。
math_bf16 和 reference head_batch4 的历史设置如实登记；不代表音频质量已通过独立评测。

登记字段见case.json：first_frame_bytes、reference_bytes以及Prompt/seed/requested_steps/num_frames。视频使用对应测试用例的完整官方参考端输出。历史sha字段不参与运行校验；大小不能证明图片内容相同，换用例时应同步替换对应参考视频及登记信息。
