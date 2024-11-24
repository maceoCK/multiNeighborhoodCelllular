import createREGL from "https://cdn.skypack.dev/regl";

// --- State management

let mouseX = window.innerWidth / 2;
let mouseY = window.innerHeight / 2;
window.addEventListener("mousemove", e => {
    mouseX = e.clientX
    mouseY = e.clientY
})
let isMouseDown = false;

window.state = {
    dt: 0.441,
    outerRadius: 8,
    ratioOfRadii: 3,
    birth1: 0.257,
    birth2: 0.479,
    survival1: 0.365,
    survival2: 0.439,
    fullness1: 0.028,
    fullness2: 0.147,

    "rr": -0.546,
    "rg": 0.295,
    "rb": 0.685,
    "gr": -0.646,
    "gg": 0.658,
    "gb": 0.552,
    "br": 0.477,
    "bg": 0.627,
    "bb": -0.532,

    brushRadius: 15,
    brushRed: 0.8,
    brushGreen: 0.9,
    brushBlue: 1,
    randomSeed: 0,
    kill: 0,
}

function setState(partial) {
    Object.entries(partial).forEach(([key, value]) => {
        state[key] = value;
    })
    const params = new URLSearchParams(state);
    window.history.replaceState({}, '', `${location.pathname}?${params}`);
}

function initStateBindings() {
    const params = Object.fromEntries(new URLSearchParams(location.search).entries());
    console.log(state, params, {...state, ...params})
    Object.entries({ ...state, ...params }).forEach(([key, value]) => {
        const boundEl = document.querySelector(`[data-state="${key}"]`);
        if (!boundEl) return;

        boundEl.value = value;
        state[key] = parseFloat(value);
        boundEl.addEventListener("input", (e) => {
            setState({
                [key]: parseFloat(e.target.value)
            })
        })
    })

    const randomSeedBtn = document.querySelector("#randomSeed");
    randomSeedBtn.addEventListener("mousedown", () => state.randomSeed = 1)
    randomSeedBtn.addEventListener("mouseup", () => state.randomSeed = 0)

    const killBtn = document.querySelector("#kill");
    killBtn.addEventListener("mousedown", () => state.kill = 1)
    killBtn.addEventListener("mouseup", () => state.kill = 0)

    window.addEventListener("mouseup", e => {
        if (e.target.tagName === "CANVAS") {
            isMouseDown = false;
        }
    });
    window.addEventListener("mousedown", e => {
        if (e.target.tagName === "CANVAS") {
            isMouseDown = true;
        }
    });
};

initStateBindings();

// --- Actual simulation

function initSimulation() {
    let width = window.innerWidth / 2;
    let height = window.innerHeight / 2;

    const regl = createREGL()

    const kernelTexture = regl.texture({
        width: state.outerRadius,
        height: state.outerRadius,
    });

    const kernelFbo = regl.framebuffer({
        color: kernelTexture,
        depthStencil: false,
    });

    const drawKernel = regl({
        vert: `
            precision highp float;
            attribute vec2 position;
            varying vec2 uv;
            void main() {
                uv = position;
                gl_Position = vec4(position, 0, 1);
            }
        `,
        frag: `
            precision highp float;
            uniform float ra;

            void main() {
                vec2 center = vec2(ra / 2.);
                float d = distance(center, gl_FragCoord.xy);
                float c = .5 * sin(d) + .5;
                gl_FragColor = vec4(vec3(c), 1.);
            }
        `,

        attributes: {
            position: [
                [-1, 1], [1, 1], [1, -1],
                [1, -1], [-1, -1], [-1, 1],
            ]
        },
        count: 6,

        uniforms: {
            ra: () => state.outerRadius,
            rr: () => state.ratioOfRadii,
        },
    });
        
    kernelFbo.use(() => {
        drawKernel();
    });

    let frame = 0;
    const drawLife = regl({
        frag: `
            precision lowp float;

            uniform float currentFrame;
            uniform vec2 resolution;
            uniform sampler2D readTexture;
            uniform float dt;
            uniform mat3 color_conv;
            uniform sampler2D kernelTexture;
            // based on <https://git.io/vz29Q>
            // ---------------------------------------------
            // smoothglider (discrete time stepping 2D)
            uniform float ra;         // outer radius
            uniform float rr;          // ratio of radii
            uniform float b1;        // birth1
            uniform float b2;        // birth2
            uniform float s1;        // survival1
            uniform float s2;        // survival2
            uniform float alpha_n;   // sigmoid width for outer fullness
            uniform float alpha_m;   // sigmoid width for inner fullness
            // ---------------------------------------------

            const vec3 h3 = vec3(.5);

            vec3 sigma1(vec3 x,vec3 a,float alpha) 
            { 
                return 1.0 / (1.0 + exp(-(x - a) * 4.0 / alpha));
            }

            vec3 sigma2(vec3 x,vec3 a,vec3 b,float alpha)
            {
                return sigma1(x,a,alpha) * (1.0 - sigma1(x,b,alpha));
            }

            vec3 sigma_m(float x,float y,vec3 m,float alpha)
            {
                return x * (1.0 - sigma1(m,h3,alpha)) 
                    + y * sigma1(m,h3,alpha);
            }

            // the transition function
            // (n = outer fullness, m = inner fullness)
            vec3 s(vec3 n,vec3 m)
            {
                return sigma2(
                    n,
                    sigma_m(b1,s1,m,alpha_m), 
                    sigma_m(b2,s2,m,alpha_m),
                    alpha_n
                );
            }

            float ramp_step(float x,float a,float ea)
            {
                return clamp((x-a)/ea + 0.5,0.0,1.0);
            }

            vec3 sum1(vec3 v) {
                return v / (v.x + v.y + v.z);
            }

            vec3 getBiasedVal(vec3 val, vec3 redBias, vec3 greenBias, vec3 blueBias) {
                return vec3(
                    redBias.x * val.x + redBias.y * val.y + redBias.z * val.z,
                    greenBias.x * val.x + greenBias.y * val.y + greenBias.z * val.z,
                    blueBias.x * val.x + blueBias.y * val.y + blueBias.z * val.z
                );
            }

            void main()
            {
                vec3 redBias = sum1(color_conv[0]);

                vec3 greenBias = sum1(color_conv[1]);
                vec3 blueBias = sum1(color_conv[2]);

                vec2 uv = gl_FragCoord.xy / resolution.xy;
                const float maxRa = 24.;
                // inner radius:
                float rb = ra/rr;
                // area of annulus:
                const float PI = 3.14159265358979;
                float AREA_OUTER = PI * (ra*ra - rb*rb);
                float AREA_INNER = PI * rb * rb;
                
                // how full are the annulus and inner disk?
                vec3 outf = vec3(0., 0., 0.), inf = vec3(0., 0., 0.);
                for(float _dx=0.; _dx<=2.*maxRa; _dx++) {
                    float dx = _dx - ra;
                    for(float _dy=0.; _dy<=2.*maxRa; _dy++) {
                        float dy = _dy - ra;
                        float r = sqrt(float(dx*dx + dy*dy));
                        vec2 txy = mod((gl_FragCoord.xy + vec2(dx,dy)) / resolution.xy, 1.);
                        vec3 val = getBiasedVal(texture2D(readTexture, txy).xyz, redBias, greenBias, blueBias); 
                        inf  += val * ramp_step(-r,-rb,1.0);
                        outf += val * ramp_step(-r,-ra,1.0) 
                                    * ramp_step(r,rb,1.0);
                        if (dy > 2. * ra) break;
                    }
                    if (dx > 2. * ra) break;
                }
                outf /= AREA_OUTER; // normalize by area
                inf /= AREA_INNER; // normalize by area
                
                vec3 prev = texture2D(readTexture, gl_FragCoord.xy / resolution.xy).xyz;
                // square dt to get a nicer UX out of the slider
                vec3 c = prev + dt*dt * (s(outf,inf) - prev);
                gl_FragColor = vec4(c,1);
            }
        `,
        vert: `
            attribute vec2 position;
            void main() {
            gl_Position = vec4(position, 0, 1);
            }
        `,
    
        attributes: {
            position: [
                [-1, 1], [1, 1], [1, -1],
                [1, -1], [-1, -1], [-1, 1],
            ]
        },

        uniforms: {
            resolution: ctx => [ctx.viewportWidth, ctx.viewportHeight],
            currentFrame: () => frame,
            readTexture: regl.prop("readTexture"),

            dt: () => state.dt,
            ra: () => state.outerRadius,
            rr: () => state.ratioOfRadii,
            b1: () => state.birth1,
            b2: () => state.birth2,
            s1: () => state.survival1,
            s2: () => state.survival2,
            alpha_n: () => state.fullness1,
            alpha_m: () => state.fullness2,

            color_conv: () => {
                return [
                    state.rr, state.rg, state.rb,
                    state.gr, state.gg, state.gb,
                    state.br, state.bg, state.bb,
                ];
            },

            kernel: kernelFbo,
        },
    
        count: 6
    })

    // render texture to screen
    const drawToCanvas = regl({
        vert: `
            precision highp float;
            attribute vec2 position;
            varying vec2 uv;
            void main() {
                uv = position;
                gl_Position = vec4(position, 0, 1);
            }
        `,
        frag: `
            precision highp float;
            uniform sampler2D readTexture;
            varying vec2 uv;

            void main () {
            vec4 color = texture2D(readTexture, uv * 0.5 + 0.5);
            gl_FragColor = color;
            }
        `,
        uniforms: {
            readTexture: regl.prop('readTexture'),
        },
        attributes: {
            position: [
                [-1, 1], [1, 1], [1, -1],
                [1, -1], [-1, -1], [-1, 1],
            ]
        },
        count: 6,
    });


    const drawInitializer = regl({
        vert: `
            precision highp float;
            attribute vec2 position;
            varying vec2 uv;
            void main() {
                uv = position;
                gl_Position = vec4(position, 0, 1);
            }
        `,
        frag: `
            precision highp float;
            uniform sampler2D readTexture;
            uniform vec3 mouse;
            uniform float brushRadius;
            uniform float currentFrame;
            uniform vec3 brushColor;
            uniform float randomSeed;
            uniform float kill;
            varying vec2 uv;


            // 1 out, 3 in... <https://www.shadertoy.com/view/4djSRW>
            #define MOD3 vec3(.1031,.11369,.13787)
            float hash13(vec3 p3) {
                p3 = fract(p3 * MOD3);
                p3 += dot(p3, p3.yzx+19.19);
                return fract((p3.x + p3.y)*p3.z);
            }

            float rand(vec2 co){
                return fract(sin(dot(co, vec2(12.9898, 78.233))) * 43758.5453);
            }

            void main () {
                vec4 color = texture2D(readTexture, uv * 0.5 + 0.5);
                if (mouse.z > 0.5 && distance(mouse.xy, gl_FragCoord.xy) < brushRadius){
                    color = vec4(brushColor, 1);
                }
                if (randomSeed > 0.5) {
                    color = vec4(
                        hash13(vec3(gl_FragCoord.xy, currentFrame)),
                        hash13(vec3(gl_FragCoord.xy, currentFrame + 1.)),
                        hash13(vec3(gl_FragCoord.xy, currentFrame + 2.)),
                        1
                    );
                }
                if (kill > 0.5) {
                    color = vec4(vec3(0.), 1.);
                }
                gl_FragColor = color;
            }
        `,
        uniforms: {
            readTexture: regl.prop('readTexture'),
            mouse: () => [
                mouseX / (window.innerWidth / width),
                (window.innerHeight - mouseY) / (window.innerHeight / height),
                isMouseDown || frame < 10 ? 1 : 0,
            ],
            currentFrame: () => frame,

            brushRadius: () => state.brushRadius,
            brushColor: () => [
                state.brushRed,
                state.brushGreen,
                state.brushBlue,
            ],

            randomSeed: () => state.randomSeed,
            kill: () => state.kill,
        },
        attributes: {
            position: [
                [-1, 1], [1, 1], [1, -1],
                [1, -1], [-1, -1], [-1, 1],
            ]
        },
        count: 6,
    });

    const textureOptions = {
        width,
        height,
        mag: "linear"
    };
    
    const createPingPongBuffers = () => {
        const tex1 = regl.texture(textureOptions);
        const tex2 = regl.texture(textureOptions);
        const one = regl.framebuffer({
            color: tex1,
            depthStencil: false
        });
        const two = regl.framebuffer({
            color: tex2,
            depthStencil: false
        });
        let flip = false
        return () => {
            flip = !flip
            return flip ? [one, two] : [two, one]
        }
    };
    const getFBOs = createPingPongBuffers();
    regl.frame(() => {
        if (isMouseDown || frame < 10 || state.randomSeed || state.kill) {
            const [read, write] = getFBOs();
            write.use(() => {
                drawInitializer({
                    readTexture: read,
                });
            });
        }
    
        const [read, write] = getFBOs();
        write.use(() => {
            drawLife({
                readTexture: read,
            });
        });
        drawToCanvas({
            readTexture: write
        });
        frame++
    })

    return () => regl.destroy();
}

let cleanup = initSimulation();

window.addEventListener("resize", () => {
    cleanup?.();
    cleanup = initSimulation();
});
