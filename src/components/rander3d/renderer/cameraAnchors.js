export const cameraAnchors = [
    {
        title: '光栅体素渲染',
        text: '光栅化体素几何，结合深度测试与片元着色生成画面。',
        camera: { x: 0.25, y: 7.5, z: 6, yaw: 0, pitch: 0 },
    },
    {
        title: '辉光 （Bloom）',
        text: '提取并模糊高亮区域，再叠加回画面，形成蜡烛光晕',
        camera: { x: 5.2, y: 5.4, z: 23.5, yaw: 0.5, pitch: 0.55 },
    },
    {
        title: 'DDA阴影',
        text: '沿光线逐格遍历体素，检测遮挡并生成阴影。',
        camera: { x: -2, y: 19, z: 59, yaw: 1.48, pitch: 1.05 },
    },
    {
        title: '环境光遮蔽（AO）',
        text: '计算体素角点的环境光可见度，强化柱头雕饰与拱顶凹槽的明暗层次。',
        camera: { x: -4.5, y: 20, z: 90, yaw: 0.28, pitch: -0.22 },
    },
];

export function cameraAtProgress(progress) {
    const position = Math.max(0, Math.min(cameraAnchors.length - 1, progress));
    const index = Math.min(cameraAnchors.length - 2, Math.floor(position));
    const amount = position - index;
    const from = cameraAnchors[index].camera, to = cameraAnchors[index + 1].camera;
    return Object.fromEntries(Object.keys(from).map((key) => {
        const delta = key === 'yaw'
            ? Math.atan2(Math.sin(to[key] - from[key]), Math.cos(to[key] - from[key]))
            : to[key] - from[key];
        return [key, from[key] + delta * amount];
    }));
}
