# FL2VA 参考视频

`reference_fl2va.mp4` 为官方 legacy 后端生成的完整参考视频，对应 `inputs/first_frame.png`、Prompt `A dog falls asleep`、seed42、50 步请求及 124 帧。

`case.json` 保存输入、生成参数、视频规格与参考端运行配置。参考端使用 math_bf16，reference_max_heads 为 4。

默认启动只生成并保存 native 视频。执行 `bash run_submission.sh check` 时，脚本生成新的 native 视频，再调用未修改的官方比较器与本目录参考视频比较；参考视频不会重新生成。

更换首帧、Prompt 或其他生成参数时，应使用对应测试用例的参考视频，并同步更新 `case.json` 中的 `first_frame_bytes`、`reference_bytes`、`prompt`、`seed`、`requested_steps` 和 `num_frames`。
