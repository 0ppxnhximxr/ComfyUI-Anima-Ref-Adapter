// Anima Ref Adapter: show only the inputs that apply to the selected adapter file.
// pilot/v2 adapters use strength, ref_guidance, correction and ref_end; SnapRef adapters use snapref_mode,
// ref_caption, snapref_guidance and ref_weight, plus the guidance inputs when snapref_guidance is on.
// The format comes from the file's metadata (GET /anima_ref_adapter/format); when it is unknown, every
// input stays visible. Hidden inputs keep their values, and the backend picks the same format on its own.
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const LEGACY = ["strength"];
const SNAPREF = ["snapref_mode", "ref_caption", "snapref_guidance", "ref_weight"];
const GUIDANCE = ["ref_guidance", "correction", "ref_end"];
const formats = new Map();

function formatOf(name) {
    if (!formats.has(name)) {
        formats.set(name, api.fetchApi(`/anima_ref_adapter/format?name=${encodeURIComponent(name)}`)
            .then((r) => (r.ok ? r.json() : {}))
            .then((j) => j.format ?? null)
            .catch(() => null));
    }
    return formats.get(name);
}

const widget = (node, name) => node.widgets?.find((w) => w.name === name);

async function refresh(node) {
    const lora = widget(node, "lora_name");
    if (!lora) return;
    const format = await formatOf(lora.value);
    const snapref = format === "snapref";
    const hide = new Set();
    if (format === "legacy") SNAPREF.forEach((n) => hide.add(n));
    if (snapref) {
        LEGACY.forEach((n) => hide.add(n));
        if (!widget(node, "snapref_guidance")?.value) GUIDANCE.forEach((n) => hide.add(n));
    }
    for (const w of node.widgets ?? []) {
        if ([...LEGACY, ...SNAPREF, ...GUIDANCE].includes(w.name)) w.hidden = hide.has(w.name);
    }
    const clip = node.inputs?.find((i) => i.name === "clip");
    if (clip) clip.label = snapref ? "clip (required)" : "clip (SnapRef only)";
    node.setSize([node.size[0], node.computeSize()[1]]);
    node.setDirtyCanvas?.(true, true);
}

app.registerExtension({
    name: "AnimaRefAdapter.FormatInputs",
    nodeCreated(node) {
        if (node.comfyClass !== "AnimaRefAdapter") return;
        for (const name of ["lora_name", "snapref_guidance"]) {
            const w = widget(node, name);
            if (!w) continue;
            const callback = w.callback;
            w.callback = function (...args) {
                const result = callback?.apply(this, args);
                refresh(node);
                return result;
            };
        }
        refresh(node);
    },
    loadedGraphNode(node) {
        // saved widget values are restored after nodeCreated
        if (node.comfyClass === "AnimaRefAdapter") refresh(node);
    },
});
