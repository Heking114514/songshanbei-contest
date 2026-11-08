# 固定 FL2VA 参考视频

reference_fl2va.mp4 是历史完整官方 legacy 参考端生成的视频，不是本轮生成的 native 视频。
case.json 绑定其图片字节、Prompt、seed42、50步请求、124帧、视频 SHA 和原始运行标识。

上传时将本目录与交付包一起放到个人 AFS。bash run_submission.sh check
会完整生成新的 native 视频，再使用本参考视频调用未修改官方比较器；reference 不会重新生成。
默认不加 check 时，只生成并保存 native 视频。

本视频只适用于 case.json 对应的测试用例。换图片或官方参数后，需要对应的新参考视频与登记信息，
重新制作交付包；不能拿本视频比较其他用例，也不能以复用参考视频的结果宣称同轮双方加速比。
math_bf16 和 reference head_batch4 的历史设置如实登记；不代表音频质量已通过独立评测。
