"""Legacy Gradio interface for NVC."""

import html
import inspect

import gradio as gr
from starlette.templating import Jinja2Templates

from gui import *


def install_gradio_template_compat():
    """Adapt Gradio 3's template call to the newer Starlette signature."""
    template_response = Jinja2Templates.TemplateResponse
    parameters = tuple(inspect.signature(template_response).parameters)
    if len(parameters) < 3 or parameters[1:3] != ("request", "name"):
        return
    if getattr(template_response, "_nvc_gradio_compat", False):
        return

    def compatible(self, *args, **kwargs):
        if len(args) >= 2 and isinstance(args[0], str) and isinstance(args[1], dict):
            name, context, *rest = args
            return template_response(
                self,
                context["request"],
                name,
                context,
                *rest,
                **kwargs,
            )
        return template_response(self, *args, **kwargs)

    compatible._nvc_gradio_compat = True
    Jinja2Templates.TemplateResponse = compatible


install_gradio_template_compat()


class ToolButton(gr.Button, gr.components.FormComponent):
    """Small emoji button that fits inside Gradio forms."""

    def __init__(self, **kwargs):
        super().__init__(variant="tool", **kwargs)

    def get_block_name(self):
        return "button"


def option_help_html(text):
    return (
        '<div class="opt-help" tabindex="0">?'
        '<div class="opt-help-popup">%s</div></div>'
        % html.escape(i18n(text)).replace("\n", "<br>")
    )


def launch_gui_with_port_fallback(app, config):
    next_port = config.listen_port
    queued_app = app.queue(concurrency_count=511, max_size=1022)
    while True:
        config.listen_port = find_available_port(next_port)
        if config.listen_port != next_port:
            logger.warning(
                "Port %s is occupied; trying port %s instead.",
                next_port,
                config.listen_port,
            )
        try:
            queued_app.launch(
                server_name="0.0.0.0",
                inbrowser=not config.noautoopen,
                server_port=config.listen_port,
                quiet=True,
            )
            return config.listen_port
        except OSError as error:
            if not str(error).startswith(f"Port {config.listen_port} is in use."):
                raise
            if config.listen_port == 65535:
                raise OSError(
                    "No available TCP port through 65535; GUI was not started."
                ) from error
            next_port = config.listen_port + 1


TRAINING_INFO_CSS = """
#training-info-step2b textarea,
#training-info-step3 textarea {
    height: 12rem !important;
    min-height: 12rem !important;
    max-height: 12rem !important;
    overflow-y: auto !important;
    resize: none !important;
}
#training-info-step2a textarea {
    height: 8rem !important;
    min-height: 8rem !important;
    max-height: 8rem !important;
    overflow-y: auto !important;
    resize: none !important;
}
.multispeaker-help {
    position: relative;
    display: inline-block;
    margin: 8px 0 0 6px;
    color: #2563eb;
    cursor: help;
    font-weight: 700;
    z-index: 10001;
}
.multispeaker-help > .multispeaker-help-popup {
    position: absolute;
    left: 18px;
    top: -8px;
    width: min(480px, calc(100vw - 80px));
    padding: 14px 16px;
    border: 1px solid #93c5fd;
    border-radius: 6px;
    background: #eff6ff;
    color: #1e3a8a;
    box-shadow: 0 10px 28px rgba(15, 23, 42, .18);
    line-height: 1.55;
    font-size: 13px;
    font-weight: 400;
    z-index: 10002;
    opacity: 0;
    pointer-events: none;
    transform: translateY(-4px);
    transition: opacity .16s ease, transform .16s ease;
}
.multispeaker-help-warning {
    margin-bottom: 8px;
    color: #b91c1c;
    font-weight: 700;
}
.multispeaker-help:hover > .multispeaker-help-popup,
.multispeaker-help:focus-within > .multispeaker-help-popup {
    opacity: 1;
    transform: translateY(0);
}
.opt-help {
    position: relative;
    display: inline-block;
    margin: 0 0 0 6px;
    color: #2563eb;
    cursor: help;
    font-weight: 700;
    z-index: 10001;
}
.opt-help > .opt-help-popup {
    position: absolute;
    left: 18px;
    top: -8px;
    width: min(480px, calc(100vw - 80px));
    padding: 14px 16px;
    border: 1px solid #93c5fd;
    border-radius: 6px;
    background: #eff6ff;
    color: #1e3a8a;
    box-shadow: 0 10px 28px rgba(15, 23, 42, .18);
    line-height: 1.55;
    font-size: 13px;
    font-weight: 400;
    white-space: pre-wrap;
    z-index: 10002;
    opacity: 0;
    pointer-events: none;
    transform: translateY(-4px);
    transition: opacity .16s ease, transform .16s ease;
}
.opt-help:hover > .opt-help-popup,
.opt-help:focus-within > .opt-help-popup {
    opacity: 1;
    transform: translateY(0);
}
.opt-help-host {
    position: relative !important;
    z-index: 10000 !important;
    overflow: visible !important;
}
#training-mode-column {
    flex: 0 0 270px !important;
    min-width: 270px !important;
    max-width: 270px !important;
}
#training-mode-selector .wrap {
    flex-wrap: nowrap !important;
}
#multispeaker-help-column {
    position: relative !important;
    z-index: 10000 !important;
    flex: 0 0 44px !important;
    min-width: 44px !important;
    max-width: 44px !important;
    overflow: visible !important;
}
#multispeaker-help-column > div {
    overflow: visible !important;
}
.multispeaker-submit-status {
    padding: 10px 12px;
    border: 1px solid;
    border-radius: 6px;
    white-space: pre-wrap;
    line-height: 1.45;
}
"""


def change_f0_method(f0method8):
    if f0method8 in ("rmvpe", "nmvpe", "fcpe"):
        visible = F0GPUVisible
    else:
        visible = False
    return {"visible": visible, "__type__": "update"}


with gr.Blocks(title="NVC GUI", css=TRAINING_INFO_CSS) as app:
    gr.Markdown("## NVC GUI")
    gr.Markdown(
        value=i18n(
            "本软件以MIT协议开源, 作者不对软件具备任何控制力, 使用软件者、传播软件导出的声音者自负全责. <br>如不认可该条款, 则不能使用或引用软件包内任何代码和文件. 详见根目录<b>LICENSE</b>."
        )
    )
    with gr.Tabs():
        with gr.TabItem(i18n("模型推理")):
            with gr.Row():
                sid0 = gr.Dropdown(label=i18n("推理音色"), choices=sorted(names))
                with gr.Column():
                    refresh_button = gr.Button(
                        i18n("刷新音色列表"), variant="primary"
                    )
                    clean_button = gr.Button(i18n("卸载音色省显存"), variant="primary")
                spk_item = gr.Slider(
                    minimum=0,
                    maximum=2333,
                    step=1,
                    label=i18n("请选择说话人id"),
                    value=0,
                    visible=False,
                    interactive=True,
                )
                spk_item_dropdown = gr.Dropdown(
                    label=i18n("选择多说话人音色"),
                    choices=[],
                    value=None,
                    visible=False,
                    interactive=True,
                )
                clean_button.click(
                    fn=clean, inputs=[], outputs=[sid0], api_name="infer_clean"
                )
            with gr.TabItem(i18n("单次推理")):
                with gr.Group():
                    with gr.Row():
                        with gr.Column():
                            with gr.Row(equal_height=True):
                                with gr.Column(scale=1, min_width=120):
                                    vc_transform0 = gr.Number(
                                        label=i18n("变调(整数, 半音数量, 升八度12降八度-12)"),
                                        value=0,
                                    )
                                with gr.Column(scale=2, min_width=200):
                                    f0method0 = gr.Radio(
                                        label=i18n("选择音高提取算法"),
                                        choices=["pm", "rmvpe", "fcpe"],
                                        value="rmvpe",
                                        interactive=True,
                                    )
                            with gr.Row():
                                dynamic_autotune0 = gr.Checkbox(label=i18n("自动匹配模型音域（仅八度）"), value=False, interactive=True)
                            with gr.Column(visible=False) as pitch_profile_controls0:
                                fallback_pitch_hz0 = gr.Slider(minimum=50, maximum=500, step=5, label=i18n("无音高配置时的目标中值（Hz）"), value=155, interactive=True)
                                pitch_profile_dataset0 = gr.Textbox(
                                    label=i18n("音高配置数据集文件或文件夹路径"),
                                    placeholder="C:\\Users\\Desktop\\voice_dataset",
                                )
                                pitch_profile_build0 = gr.Button(
                                    i18n("从数据集创建音高配置"), variant="primary"
                                )
                                pitch_profile_status0 = gr.Textbox(
                                    label=i18n("音高配置状态"), interactive=False
                                )
                            dynamic_autotune0.change(
                                toggle_pitch_profile_controls,
                                inputs=[dynamic_autotune0],
                                outputs=[pitch_profile_controls0],
                                queue=False,
                            )
                            pitch_profile_build0.click(
                                create_model_pitch_profile_status,
                                inputs=[sid0, pitch_profile_dataset0],
                                outputs=[pitch_profile_status0],
                            )
                            input_audio0 = gr.Audio(
                                label=i18n("拖拽或点击上传待处理音频"),
                                source="upload",
                                type="filepath",
                                interactive=True,
                            )

                        with gr.Column():
                            resample_sr0 = gr.Slider(
                                minimum=0,
                                maximum=48000,
                                label=i18n("后处理重采样至最终采样率，0为不进行重采样"),
                                value=0,
                                step=1,
                                interactive=True,
                            )
                            rms_mix_rate0 = gr.Slider(
                                minimum=0,
                                maximum=1,
                                label=i18n(
                                    "输入源音量包络替换输出音量包络融合比例，越靠近1越使用输出包络"
                                ),
                                value=0.25,
                                interactive=True,
                            )
                            protect0 = gr.Slider(
                                minimum=0,
                                maximum=0.5,
                                label=i18n(
                                    "保护清辅音和呼吸声，防止电音撕裂等artifact，拉满0.5不开启，调低加大保护力度但可能降低索引效果"
                                ),
                                value=0.33,
                                step=0.01,
                                interactive=True,
                            )
                            index_rate1 = gr.Slider(
                                minimum=0,
                                maximum=1,
                                label=i18n("检索特征占比"),
                                value=0.75,
                                interactive=True,
                            )
                            file_index1 = gr.Textbox(
                                label=i18n("特征检索库文件路径（选择模型后自动匹配，可手动修改）"),
                                placeholder="C:\\Users\\Desktop\\model_example.index",
                                interactive=True,
                            )
                            refresh_button.click(
                                fn=change_choices,
                                inputs=[],
                                outputs=sid0,
                                api_name="infer_refresh",
                            )
                with gr.Group():
                    with gr.Column():
                        but0 = gr.Button(i18n("转换"), variant="primary")
                        with gr.Row():
                            vc_output1 = gr.Textbox(label=i18n("输出信息"))
                            vc_output2 = gr.Audio(
                                label=i18n("输出音频(右下角三个点,点了可以下载)")
                            )

                        but0.click(
                            report_missing_index,
                            [file_index1],
                            [],
                            queue=False,
                            api_name="infer_check_index",
                        )
                        but0.click(
                            vc_single_with_speaker,
                            [
                                spk_item,
                                spk_item_dropdown,
                                input_audio0,
                                vc_transform0,
                                f0method0,
                                file_index1,
                                index_rate1,
                                resample_sr0,
                                rms_mix_rate0,
                                protect0,
                                dynamic_autotune0,
                                fallback_pitch_hz0,
                            ],
                            [vc_output1, vc_output2],
                            api_name="infer_convert",
                        )
            with gr.TabItem(i18n("批量推理")):
                gr.Markdown(
                    value=i18n(
                        "批量转换, 输入待转换音频文件夹, 或上传多个音频文件, 在指定文件夹(默认opt)下输出转换的音频. "
                    )
                )
                with gr.Row():
                    with gr.Column():
                        vc_transform1 = gr.Number(
                            label=i18n("变调(整数, 半音数量, 升八度12降八度-12)"),
                            value=0,
                        )
                        opt_input = gr.Textbox(
                            label=i18n("指定输出文件夹"), value="opt"
                        )
                        file_index3 = gr.Textbox(
                            label=i18n("特征检索库文件路径（选择模型后自动匹配，可手动修改）"),
                            value="",
                            interactive=True,
                        )
                        f0method1 = gr.Radio(
                            label=i18n("选择音高提取算法"),
                            choices=["pm", "rmvpe", "fcpe"],
                            value="rmvpe",
                            interactive=True,
                        )
                        with gr.Row():
                            dynamic_autotune1 = gr.Checkbox(label=i18n("自动匹配模型音域（仅八度）"), value=False, interactive=True)
                        with gr.Column(visible=False) as pitch_profile_controls1:
                            fallback_pitch_hz1 = gr.Slider(minimum=50, maximum=500, step=5, label=i18n("无音高配置时的目标中值（Hz）"), value=155, interactive=True)
                            pitch_profile_dataset1 = gr.Textbox(
                                label=i18n("音高配置数据集文件或文件夹路径"),
                                placeholder="C:\\Users\\Desktop\\voice_dataset",
                            )
                            pitch_profile_build1 = gr.Button(
                                i18n("从数据集创建音高配置"), variant="primary"
                            )
                            pitch_profile_status1 = gr.Textbox(
                                label=i18n("音高配置状态"), interactive=False
                            )
                        dynamic_autotune1.change(
                            toggle_pitch_profile_controls,
                            inputs=[dynamic_autotune1],
                            outputs=[pitch_profile_controls1],
                            queue=False,
                        )
                        pitch_profile_build1.click(
                            create_model_pitch_profile_status,
                            inputs=[sid0, pitch_profile_dataset1],
                            outputs=[pitch_profile_status1],
                        )
                        format1 = gr.Radio(
                            label=i18n("导出文件格式"),
                            choices=["wav", "flac", "mp3", "m4a"],
                            value="wav",
                            interactive=True,
                        )

                    with gr.Column():
                        resample_sr1 = gr.Slider(
                            minimum=0,
                            maximum=48000,
                            label=i18n("后处理重采样至最终采样率，0为不进行重采样"),
                            value=0,
                            step=1,
                            interactive=True,
                        )
                        rms_mix_rate1 = gr.Slider(
                            minimum=0,
                            maximum=1,
                            label=i18n(
                                "输入源音量包络替换输出音量包络融合比例，越靠近1越使用输出包络"
                            ),
                            value=1,
                            interactive=True,
                        )
                        protect1 = gr.Slider(
                            minimum=0,
                            maximum=0.5,
                            label=i18n(
                                "保护清辅音和呼吸声，防止电音撕裂等artifact，拉满0.5不开启，调低加大保护力度但可能降低索引效果"
                            ),
                            value=0.33,
                            step=0.01,
                            interactive=True,
                        )
                        index_rate2 = gr.Slider(
                            minimum=0,
                            maximum=1,
                            label=i18n("检索特征占比"),
                            value=1,
                            interactive=True,
                        )
                with gr.Row():
                    dir_input = gr.Textbox(
                        label=i18n(
                            "输入待处理音频文件夹路径(去文件管理器地址栏拷就行了)"
                        ),
                        placeholder="C:\\Users\\Desktop\\input_vocal_dir",
                    )
                    inputs = gr.File(
                        file_count="multiple",
                        label=i18n("也可批量输入音频文件, 二选一, 优先读文件夹"),
                    )

                with gr.Row():
                    but1 = gr.Button(i18n("转换"), variant="primary")
                    vc_output3 = gr.Textbox(label=i18n("输出信息"))

                    but1.click(
                        report_missing_index,
                        [file_index3],
                        [],
                        queue=False,
                        api_name="infer_check_index_batch",
                    )
                    but1.click(
                        vc_multi_with_speaker,
                        [
                            spk_item,
                            spk_item_dropdown,
                            dir_input,
                            opt_input,
                            inputs,
                            vc_transform1,
                            f0method1,
                            file_index3,
                            index_rate2,
                            resample_sr1,
                            rms_mix_rate1,
                            protect1,
                            format1,
                            dynamic_autotune1,
                            fallback_pitch_hz1,
                        ],
                        [vc_output3],
                        api_name="infer_convert_batch",
                    )
                sid0.change(
                    fn=vc.get_vc,
                    inputs=[sid0, protect0, protect1],
                    outputs=[
                        spk_item,
                        spk_item_dropdown,
                        protect0,
                        protect1,
                        file_index1,
                        file_index3,
                    ],
                    api_name="infer_change_voice",
                )
                spk_item.change(
                    fn=update_speaker_index,
                    inputs=[sid0, spk_item, spk_item_dropdown],
                    outputs=[file_index1, file_index3],
                    queue=False,
                    api_name="infer_change_speaker_index_slider",
                )
                spk_item_dropdown.change(
                    fn=update_dropdown_speaker_index,
                    inputs=[sid0, spk_item_dropdown],
                    outputs=[file_index1, file_index3],
                    queue=False,
                    api_name="infer_change_speaker_index_dropdown",
                )
        with gr.TabItem(i18n("人声伴奏分离&去混响")):
            with gr.Group():
                gr.Markdown(
                    value=i18n(
                        "人声、伴奏与混响批量处理，使用pymss/MSST模型。"
                    )
                )
                with gr.Row():
                    with gr.Column():
                        dir_wav_input = gr.Textbox(
                            label=i18n("输入待处理音频文件夹路径"),
                            placeholder="C:\\Users\\Desktop\\todo-songs",
                        )
                        wav_inputs = gr.File(
                            file_count="multiple",
                            label=i18n("也可批量输入音频文件, 二选一, 优先读文件夹"),
                        )
                    with gr.Column():
                        model_choose = gr.Dropdown(
                            label=i18n("处理方式"),
                            choices=pymss_names,
                            value=pymss_names[0],
                            interactive=True,
                        )
                        model_info = gr.Textbox(
                            label=i18n("底层模型"),
                            value=get_model_info(pymss_names[0]),
                            interactive=False,
                        )
                        model_choose.change(
                            get_model_info,
                            [model_choose],
                            [model_info],
                            queue=False,
                        )
                        opt_vocal_root = gr.Textbox(
                            label=i18n("主结果文件夹"), value="opt"
                        )
                        opt_ins_root = gr.Textbox(
                            label=i18n("分离残余文件夹"), value="opt"
                        )
                        format0 = gr.Radio(
                            label=i18n("导出文件格式"),
                            choices=["wav", "flac", "mp3", "m4a"],
                            value="flac",
                            interactive=True,
                        )
                    with gr.Row():
                        but2 = gr.Button(i18n("转换"), variant="primary")
                        stop_pymss_button = gr.Button(
                            i18n("停止分离"), variant="stop", visible=False
                        )
                    pymss_progress = gr.HTML(
                        value=render_pymss_progress(0, "等待开始", "idle")
                    )
                    vc_output4 = gr.Textbox(label=i18n("输出信息"))
                    but2.click(
                        run_pymss_separation,
                        [
                            model_choose,
                            dir_wav_input,
                            opt_vocal_root,
                            wav_inputs,
                            opt_ins_root,
                            format0,
                        ],
                        [vc_output4, pymss_progress, but2, stop_pymss_button],
                        api_name="pymss_separate",
                    )
                    stop_pymss_button.click(
                        stop_pymss_gui,
                        [],
                        [vc_output4, pymss_progress, but2, stop_pymss_button],
                        queue=False,
                    )
        with gr.TabItem(i18n("训练")):
            gr.Markdown(
                value=i18n(
                    "step1: 填写实验配置. 实验数据放在logs下, 每个实验一个文件夹, 需手工输入实验名路径, 内含实验配置, 日志, 训练得到的模型文件. "
                )
            )
            with gr.Row():
                exp_dir1 = gr.Textbox(label=i18n("输入实验名，例如：test"))
                sr2 = gr.Radio(
                    label=i18n("目标采样率"),
                    choices=["32k", "40k", "48k"],
                    value="40k",
                    interactive=True,
                )
                if_f0_3 = gr.Radio(
                    label=i18n("模型是否带音高指导(唱歌一定要, 语音可以不要)"),
                    choices=[True, False],
                    value=True,
                    interactive=True,
                )
                np7 = gr.Slider(
                    minimum=0,
                    maximum=config.n_cpu,
                    step=1,
                    label=i18n("提取音高和处理数据使用的CPU进程数"),
                    value=int(np.ceil(config.n_cpu / 1.5)),
                    interactive=True,
                )
            with gr.Group():
                gr.Markdown(
                    value=i18n(
                        "step2a: 扫描训练音频并进行切片归一化，在实验目录下生成训练wav文件。"
                    )
                )
                with gr.Row(equal_height=False):
                    with gr.Column(scale=5, min_width=420):
                        with gr.Row():
                            with gr.Column(
                                scale=0,
                                min_width=270,
                                elem_id="training-mode-column",
                            ):
                                training_mode = gr.Radio(
                                    label=i18n("训练集类型"),
                                    choices=[i18n("单说话人"), i18n("多说话人")],
                                    value=i18n("单说话人"),
                                    interactive=True,
                                    elem_id="training-mode-selector",
                                )
                            with gr.Column(
                                scale=0,
                                min_width=44,
                                elem_id="multispeaker-help-column",
                            ):
                                multispeaker_help = gr.HTML(
                                    value=(
                                        '<div class="multispeaker-help" tabindex="0">?'
                                        '<div class="multispeaker-help-popup">'
                                        '<div class="multispeaker-help-warning">%s</div>%s'
                                        "</div></div>"
                                        % (
                                            html.escape(
                                                i18n(
                                                    "注意：多说话人训练音色还原度不一定有单说话人分开训练好！"
                                                )
                                            ),
                                            html.escape(
                                                i18n(
                                                    "多说话人总文件夹只扫描根目录下的直接子文件夹，根目录文件会被忽略。\n子文件夹必须命名为x_y_z：x是说话人名称，y是说话人ID（0~109，共110个），z是训练集重复次数。\n也可以到右侧“多说话人训练集辅助”编辑并提交训练集清单。"
                                                )
                                            ).replace("\n", "<br>"),
                                        )
                                    )
                                )
                        with gr.Row():
                            trainset_dir4 = gr.Textbox(
                                label=i18n(
                                    "输入训练文件夹路径，例如：E:\\我的训练集"
                                ),
                            )
                            spk_id5 = gr.Slider(
                                minimum=0,
                                maximum=109,
                                step=1,
                                label=i18n("请指定说话人id"),
                                value=0,
                                interactive=True,
                            )
                        with gr.Row(elem_classes="opt-help-host"):
                            noise_reduce0 = gr.Checkbox(
                                label=i18n("预处理阶段启用降噪"),
                                value=False,
                                interactive=True,
                            )
                            gr.HTML(
                                value=option_help_html(
                                    "启用后使用 noisereduce 库在切片前对每段音频做降噪，"
                                    "对含底噪/环境噪声的训练集有用。\n"
                                    "干净录音不需要开启，过强的降噪会损伤语音细节。"
                                )
                            )
                            noise_reduce_strength0 = gr.Slider(
                                minimum=0.0,
                                maximum=1.0,
                                step=0.05,
                                label=i18n("降噪强度"),
                                value=0.75,
                                interactive=True,
                            )
                            gr.HTML(
                                value=option_help_html(
                                    "降噪强度 0..1：0 = 不降噪，1 = 完全消除噪声。"
                                    "推荐 0.5~0.85，过高会损失有效语音能量。"
                                )
                            )
                    with gr.Column(scale=1, min_width=150):
                        but1 = gr.Button(i18n("处理数据"), variant="primary")
                        stop_but1 = gr.Button(
                            i18n("停止处理数据"), variant="stop", visible=False
                        )
                    with gr.Column(
                        scale=3,
                        min_width=280,
                        elem_id="step2a-output-column",
                    ):
                        info1 = gr.Textbox(
                            label=i18n("输出信息"),
                            value="",
                            lines=5,
                            max_lines=5,
                            elem_id="training-info-step2a",
                        )
                but1.click(
                    preprocess_dataset,
                    [
                        trainset_dir4,
                        exp_dir1,
                        sr2,
                        np7,
                        training_mode,
                        noise_reduce0,
                        noise_reduce_strength0,
                    ],
                    [info1, but1, stop_but1],
                    api_name="train_preprocess",
                )
                stop_but1.click(
                    stop_preprocess_dataset,
                    [],
                    [info1, but1, stop_but1],
                    queue=False,
                )
                training_mode.change(
                    change_training_mode,
                    [training_mode],
                    [trainset_dir4, spk_id5],
                    queue=False,
                )
            with gr.Group():
                gr.Markdown(
                    value=i18n(
                        "step2b: 音高与hubert语义特征提取"
                    )
                )
                with gr.Row(equal_height=False):
                    with gr.Column(scale=3, min_width=320):
                        gpus6 = gr.Textbox(
                            label=i18n(
                                "hubert:以-分隔输入使用的卡号, 例如 0-1-2 使用卡0和卡1和卡2"
                            ),
                            value=feature_gpus,
                            interactive=True,
                            visible=F0GPUVisible,
                        )
                        gpus_rmvpe = gr.Textbox(
                            label=i18n(
                                "rmvpe卡号配置：以-分隔输入使用的不同进程卡号,例如0-0-1使用在卡0上跑2个进程并在卡1上跑1个进程"
                            ),
                            value=feature_gpus,
                            interactive=True,
                            visible=F0GPUVisible,
                        )
                    with gr.Column(scale=2, min_width=260):
                        gpu_info9 = gr.Textbox(
                            label=i18n("显卡信息"), value=gpu_info, visible=F0GPUVisible
                        )
                        f0method8 = gr.Radio(
                            label=i18n("选择音高提取算法"),
                            choices=["pm", "rmvpe", "fcpe"],
                            value=default_training_f0_method,
                            interactive=True,
                        )
                        embedder9 = gr.Dropdown(
                            label=i18n("选择特征嵌入模型"),
                            choices=list(EMBEDDER_CHOICES),
                            value="contentvec",
                            interactive=True,
                        )
                    with gr.Column(scale=1, min_width=150):
                        but2 = gr.Button(i18n("特征提取"), variant="primary")
                        stop_but2 = gr.Button(
                            i18n("停止特征提取"), variant="stop", visible=False
                        )
                    with gr.Column(
                        scale=3,
                        min_width=280,
                        elem_id="step2b-output-column",
                    ):
                        info2 = gr.Textbox(
                            label=i18n("输出信息"),
                            value="",
                            lines=8,
                            max_lines=8,
                            elem_id="training-info-step2b",
                        )
                f0method8.change(
                    fn=change_f0_method,
                    inputs=[f0method8],
                    outputs=[gpus_rmvpe],
                )
                but2.click(
                    extract_f0_feature,
                    [
                        gpus6,
                        np7,
                        f0method8,
                        if_f0_3,
                        exp_dir1,
                        gpus_rmvpe,
                        embedder9,
                    ],
                    [info2, but2, stop_but2],
                    api_name="train_extract_f0_feature",
                )
                stop_but2.click(
                    stop_extract_f0_feature,
                    [],
                    [info2, but2, stop_but2],
                    queue=False,
                )
            with gr.Group():
                gr.Markdown(value=i18n("step3: 填写训练设置, 开始训练模型和索引"))
                with gr.Row():
                    save_epoch10 = gr.Slider(
                        minimum=1,
                        maximum=50,
                        step=1,
                        label=i18n("保存频率save_every_epoch"),
                        value=25,
                        interactive=True,
                    )
                    total_epoch11 = gr.Slider(
                        minimum=2,
                        maximum=1200,
                        step=1,
                        label=i18n("总训练轮数total_epoch"),
                        value=500,
                        interactive=True,
                    )
                    batch_size12 = gr.Slider(
                        minimum=1,
                        maximum=40,
                        step=1,
                        label=i18n("每张显卡的batch_size"),
                        value=default_batch_size,
                        interactive=True,
                    )
                    if_save_latest13 = gr.Radio(
                        label=i18n("是否仅保存最新的ckpt文件以节省硬盘空间"),
                        choices=[i18n("是"), i18n("否")],
                        value=i18n("否"),
                        interactive=True,
                    )
                    if_cache_gpu17 = gr.Radio(
                        label=i18n(
                            "是否缓存所有训练集至显存. 10min以下小数据可缓存以加速训练, 大数据缓存会炸显存也加不了多少速"
                        ),
                        choices=[i18n("是"), i18n("否")],
                        value=i18n("否"),
                        interactive=True,
                    )
                    if_save_every_weights18 = gr.Radio(
                        label=i18n(
                            "是否在每次保存时间点将最终小模型保存至weights文件夹"
                        ),
                        choices=[i18n("是"), i18n("否")],
                        value=i18n("否"),
                        interactive=True,
                    )
                with gr.Row(elem_classes="opt-help-host"):
                    ms_mel_train0 = gr.Checkbox(
                        label=i18n("多尺度Mel损失"),
                        value=True,
                        interactive=True,
                    )
                    gr.HTML(
                        value=option_help_html(
                            "多尺度Mel损失：在多个窗长上同时算Mel-L1，通常能稍微提升合成质量，"
                            "训练变慢约5%。仅对带音高模型生效。\n来源于Applio。"
                        )
                    )
                    grad_ckpt_train0 = gr.Checkbox(
                        label=i18n("梯度检查点 (省显存)"),
                        value=False,
                        interactive=True,
                    )
                    gr.HTML(
                        value=option_help_html(
                            "梯度检查点：在前向时不保留中间激活，反向时重算，"
                            "可显著降低显存占用（8GB卡可开更大batch），代价是训练慢约15~20%。"
                            "仅对带音高模型生效。\n来源于Applio。"
                        )
                    )
                    bf16_train0 = gr.Checkbox(
                        label=i18n("BF16混合精度"),
                        value=False,
                        interactive=True,
                    )
                    gr.HTML(
                        value=option_help_html(
                            "BF16混合精度：在Ampere及更新GPU上比FP16更稳定，"
                            "无需GradScaler，配合AnyPrecisionAdamW（Kahan求和）可达FP32质量。"
                            "不支持时自动回退到原dtype。\n来源于Applio。"
                        )
                    )
                with gr.Row():
                    pretrained_G14 = gr.Textbox(
                        label=i18n("加载预训练底模G路径"),
                        value="assets/pretrained_v2/f0G40k.pth",
                        interactive=True,
                    )
                    pretrained_D15 = gr.Textbox(
                        label=i18n("加载预训练底模D路径"),
                        value="assets/pretrained_v2/f0D40k.pth",
                        interactive=True,
                    )
                    sr2.change(
                        change_sr2,
                        [sr2, if_f0_3],
                        [pretrained_G14, pretrained_D15],
                    )
                    if_f0_3.change(
                        change_f0,
                        [if_f0_3, sr2],
                        [f0method8, gpus_rmvpe, pretrained_G14, pretrained_D15],
                    )
                    gpus16 = gr.Textbox(
                        label=i18n(
                            "以-分隔输入使用的卡号, 例如   0-1-2   使用卡0和卡1和卡2"
                        ),
                        value=gpus,
                        interactive=True,
                    )
                    but3 = gr.Button(i18n("训练模型"), variant="primary")
                    stop_but3 = gr.Button(
                        i18n("停止训练模型"), variant="stop", visible=False
                    )
                    but4 = gr.Button(i18n("训练特征索引"), variant="primary")
                    stop_but4 = gr.Button(
                        i18n("停止训练索引"), variant="stop", visible=False
                    )
                    but5 = gr.Button(i18n("一键训练"), variant="primary")
                    stop_but5 = gr.Button(
                        i18n("停止一键训练"), variant="stop", visible=False
                    )
                    info3 = gr.Textbox(
                        label=i18n("输出信息"),
                        value="",
                        lines=8,
                        max_lines=8,
                        elem_id="training-info-step3",
                    )
                    but3.click(
                        click_train,
                        [
                            exp_dir1,
                            sr2,
                            if_f0_3,
                            spk_id5,
                            save_epoch10,
                            total_epoch11,
                            batch_size12,
                            if_save_latest13,
                            pretrained_G14,
                            pretrained_D15,
                            gpus16,
                            if_cache_gpu17,
                            if_save_every_weights18,
                            training_mode,
                            embedder9,
                            ms_mel_train0,
                            grad_ckpt_train0,
                            bf16_train0,
                        ],
                        [info3, but3, stop_but3, sid0],
                        api_name="train_start",
                    )
                    stop_but3.click(
                        stop_train_model,
                        [],
                        [info3, but3, stop_but3],
                        queue=False,
                    )
                    but4.click(
                        train_index,
                        [exp_dir1, training_mode],
                        [info3, but4, stop_but4],
                    )
                    stop_but4.click(
                        stop_train_index,
                        [],
                        [info3, but4, stop_but4],
                        queue=False,
                    )
                    but5.click(
                        train1key,
                        [
                            exp_dir1,
                            sr2,
                            if_f0_3,
                            trainset_dir4,
                            spk_id5,
                            np7,
                            f0method8,
                            save_epoch10,
                            total_epoch11,
                            batch_size12,
                            if_save_latest13,
                            pretrained_G14,
                            pretrained_D15,
                            gpus16,
                            if_cache_gpu17,
                            if_save_every_weights18,
                            gpus_rmvpe,
                            training_mode,
                            embedder9,
                            noise_reduce0,
                            noise_reduce_strength0,
                            ms_mel_train0,
                            grad_ckpt_train0,
                            bf16_train0,
                        ],
                        [info3, but5, stop_but5, sid0],
                        api_name="train_start_all",
                    )
                    stop_but5.click(
                        stop_train1key,
                        [],
                        [info3, but5, stop_but5],
                        queue=False,
                    )

        with gr.TabItem(i18n("多说话人训练集辅助")):
            helper_rows_state = gr.State(empty_multispeaker_rows())
            helper_row_count = gr.State(2)
            helper_page = gr.State(0)
            gr.Markdown(
                value=i18n(
                    "为多说话人训练集建立清单。空行会被忽略；路径、说话人名称、说话人ID或重复次数填写不全的行会在提交时提示。"
                )
            )
            with gr.Row():
                helper_exp_name = gr.Textbox(
                    label=i18n("输入实验名，例如：test")
                )
                helper_submit = gr.Button(i18n("提交训练集清单"), variant="primary")
                helper_previous = gr.Button(i18n("上一页"))
                helper_page_label = gr.Markdown(
                    value=i18n("第%s/%s页，共%s行") % (1, 1, 2)
                )
                helper_next = gr.Button(i18n("下一页"))
            with gr.Row():
                helper_add = gr.Button(i18n("新增一行"))
                helper_remove = gr.Button(i18n("删除末行"))
            helper_row_outputs = []
            helper_row_values = []
            for helper_slot in range(MULTISPEAKER_PAGE_SIZE):
                with gr.Row():
                    helper_path = gr.Textbox(
                        label=i18n("训练集子音频文件夹目录路径"),
                        value="",
                        visible=helper_slot < 2,
                    )
                    helper_speaker_name = gr.Textbox(
                        label=i18n("说话人名称"),
                        value="",
                        visible=helper_slot < 2,
                    )
                    helper_speaker_id = gr.Number(
                        label=i18n("说话人ID（0~109）"),
                        value=helper_slot if helper_slot < 2 else None,
                        precision=0,
                        visible=helper_slot < 2,
                    )
                    helper_repeat = gr.Number(
                        label=i18n("重复次数"),
                        value=1 if helper_slot < 2 else None,
                        precision=0,
                        visible=helper_slot < 2,
                    )
                helper_row_outputs.extend(
                    [
                        helper_path,
                        helper_speaker_name,
                        helper_speaker_id,
                        helper_repeat,
                    ]
                )
                helper_row_values.extend(
                    [helper_path, helper_speaker_name, helper_speaker_id, helper_repeat]
                )
            helper_status = gr.HTML(value="")
            helper_event_inputs = [
                helper_rows_state,
                helper_row_count,
                helper_page,
            ] + helper_row_values
            helper_event_outputs = [
                helper_rows_state,
                helper_row_count,
                helper_page,
                helper_page_label,
            ] + helper_row_outputs
            helper_add.click(
                add_multispeaker_row,
                helper_event_inputs,
                helper_event_outputs,
                queue=False,
            )
            helper_remove.click(
                remove_multispeaker_row,
                helper_event_inputs,
                helper_event_outputs,
                queue=False,
            )
            helper_previous.click(
                previous_multispeaker_page,
                helper_event_inputs,
                helper_event_outputs,
                queue=False,
            )
            helper_next.click(
                next_multispeaker_page,
                helper_event_inputs,
                helper_event_outputs,
                queue=False,
            )
            helper_submit.click(
                submit_multispeaker_rows,
                [helper_exp_name] + helper_event_inputs,
                [helper_rows_state, helper_status],
                api_name="multispeaker_manifest_submit",
            )
            exp_dir1.change(
                sync_exp_name,
                [exp_dir1, helper_exp_name],
                [helper_exp_name],
                queue=False,
            )
            helper_exp_name.change(
                sync_exp_name,
                [helper_exp_name, exp_dir1],
                [exp_dir1],
                queue=False,
            )

        with gr.TabItem(i18n("ckpt处理")):
            with gr.Group():
                gr.Markdown(value=i18n("模型融合, 可用于测试音色融合"))
                with gr.Row():
                    ckpt_a = gr.Textbox(
                        label=i18n("A模型路径"), value="", interactive=True
                    )
                    ckpt_b = gr.Textbox(
                        label=i18n("B模型路径"), value="", interactive=True
                    )
                    alpha_a = gr.Slider(
                        minimum=0,
                        maximum=1,
                        label=i18n("A模型权重"),
                        value=0.5,
                        interactive=True,
                    )
                with gr.Row():
                    sr_ = gr.Radio(
                        label=i18n("目标采样率"),
                        choices=["32k", "40k", "48k"],
                        value="40k",
                        interactive=True,
                    )
                    if_f0_ = gr.Radio(
                        label=i18n("模型是否带音高指导"),
                        choices=[i18n("是"), i18n("否")],
                        value=i18n("是"),
                        interactive=True,
                    )
                    info__ = gr.Textbox(
                        label=i18n("要置入的模型信息"),
                        value="",
                        max_lines=8,
                        interactive=True,
                    )
                    name_to_save0 = gr.Textbox(
                        label=i18n("保存的模型名不带后缀"),
                        value="",
                        max_lines=1,
                        interactive=True,
                    )
                with gr.Row():
                    but6 = gr.Button(i18n("融合"), variant="primary")
                    info4 = gr.Textbox(label=i18n("输出信息"), value="", max_lines=8)
                but6.click(
                    merge,
                    [
                        ckpt_a,
                        ckpt_b,
                        alpha_a,
                        sr_,
                        if_f0_,
                        info__,
                        name_to_save0,
                    ],
                    info4,
                    api_name="ckpt_merge",
                )  # def merge(path1,path2,alpha1,sr,f0,info):
            with gr.Group():
                gr.Markdown(
                    value=i18n("修改模型信息(仅支持weights文件夹下提取的小模型文件)")
                )
                with gr.Row():
                    ckpt_path0 = gr.Textbox(
                        label=i18n("模型路径"), value="", interactive=True
                    )
                    info_ = gr.Textbox(
                        label=i18n("要改的模型信息"),
                        value="",
                        max_lines=8,
                        interactive=True,
                    )
                    name_to_save1 = gr.Textbox(
                        label=i18n("保存的文件名, 默认空为和源文件同名"),
                        value="",
                        max_lines=8,
                        interactive=True,
                    )
                with gr.Row():
                    but7 = gr.Button(i18n("修改"), variant="primary")
                    info5 = gr.Textbox(label=i18n("输出信息"), value="", max_lines=8)
                but7.click(
                    change_info,
                    [ckpt_path0, info_, name_to_save1],
                    info5,
                    api_name="ckpt_modify",
                )
            with gr.Group():
                gr.Markdown(
                    value=i18n("查看模型信息(仅支持weights文件夹下提取的小模型文件)")
                )
                with gr.Row():
                    ckpt_path1 = gr.Textbox(
                        label=i18n("模型路径"), value="", interactive=True
                    )
                    but8 = gr.Button(i18n("查看"), variant="primary")
                    info6 = gr.Textbox(label=i18n("输出信息"), value="", max_lines=8)
                but8.click(show_info, [ckpt_path1], info6, api_name="ckpt_show")
            with gr.Group():
                gr.Markdown(
                    value=i18n(
                        "模型提取(输入logs文件夹下大文件模型路径),适用于训一半不想训了模型没有自动提取保存小文件模型,或者想测试中间模型的情况"
                    )
                )
                with gr.Row():
                    ckpt_path2 = gr.Textbox(
                        label=i18n("模型路径"),
                        value="E:\\codes\\py39\\logs\\mi-test_f0_48k\\G_23333.pth",
                        interactive=True,
                    )
                    save_name = gr.Textbox(
                        label=i18n("保存名"), value="", interactive=True
                    )
                    sr__ = gr.Radio(
                        label=i18n("目标采样率"),
                        choices=["32k", "40k", "48k"],
                        value="40k",
                        interactive=True,
                    )
                    if_f0__ = gr.Radio(
                        label=i18n("模型是否带音高指导,1是0否"),
                        choices=["1", "0"],
                        value="1",
                        interactive=True,
                    )
                    info___ = gr.Textbox(
                        label=i18n("要置入的模型信息"),
                        value="",
                        max_lines=8,
                        interactive=True,
                    )
                    but9 = gr.Button(i18n("提取"), variant="primary")
                    info7 = gr.Textbox(label=i18n("输出信息"), value="", max_lines=8)
                    ckpt_path2.change(change_info_, [ckpt_path2], [sr__, if_f0__])
                but9.click(
                    extract_small_model,
                    [ckpt_path2, save_name, sr__, if_f0__, info___],
                    info7,
                    api_name="ckpt_extract",
                )

        tab_faq = i18n("常见问题解答")
        with gr.TabItem(tab_faq):
            try:
                if tab_faq == "常见问题解答":
                    info = read_text("docs/cn/faq.md")
                else:
                    info = read_text("docs/en/faq_en.md")
                gr.Markdown(value=info)
            except Exception:
                gr.Markdown(traceback.format_exc())

def main():
    if config.iscolab:
        try:
            app.queue(concurrency_count=511, max_size=1022).launch(
                share=True,
                server_name="0.0.0.0",
                server_port=config.listen_port,
                inbrowser=False,
                prevent_thread_lock=False,
            )
        except Exception:
            traceback.print_exc()
            raise
    else:
        launch_gui_with_port_fallback(app, config)


if __name__ == "__main__":
    main()
