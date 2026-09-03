import { useEffect, useMemo, useRef } from "react";
import * as THREE from "three";

/**
 * GLSL thermal shaders ported verbatim from the preserved thermal viewer
 * (`fluxion-tauri/src-tauri/src/index.html`, issue #3249): a 5-stop colormap
 * with a simple headlight term. The `uThermalEnabled` uniform of the original
 * is unnecessary here — React swaps between standard and thermal materials.
 */
export const THERMAL_VERTEX_SHADER = /* glsl */ `
    varying vec3 vNormal;

    void main() {
        vNormal = normalize(normalMatrix * normal);
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    }
`;

export const THERMAL_FRAGMENT_SHADER = /* glsl */ `
    uniform float uTemperature;
    uniform float uMinTemp;
    uniform float uMaxTemp;

    varying vec3 vNormal;

    vec3 thermalColormap(float t) {
        t = clamp(t, 0.0, 1.0);

        vec3 cold = vec3(0.23, 0.51, 0.96);
        vec3 cool = vec3(0.13, 0.77, 0.37);
        vec3 neutral = vec3(0.92, 0.75, 0.03);
        vec3 warm = vec3(0.98, 0.45, 0.09);
        vec3 hot = vec3(0.94, 0.24, 0.24);

        if (t < 0.25) {
            return mix(cold, cool, t * 4.0);
        } else if (t < 0.5) {
            return mix(cool, neutral, (t - 0.25) * 4.0);
        } else if (t < 0.75) {
            return mix(neutral, warm, (t - 0.5) * 4.0);
        } else {
            return mix(warm, hot, (t - 0.75) * 4.0);
        }
    }

    void main() {
        float normalizedTemp = (uTemperature - uMinTemp) / max(uMaxTemp - uMinTemp, 0.001);
        vec3 color = thermalColormap(normalizedTemp);

        float lighting = 0.5 + 0.5 * dot(vNormal, normalize(vec3(1.0, 1.0, 1.0)));
        color *= lighting;

        gl_FragColor = vec4(color, 0.85);
    }
`;

interface ThermalMaterialProps {
  temperature: number;
  minTemp: number;
  maxTemp: number;
  wireframe: boolean;
}

export function ThermalMaterial({
  temperature,
  minTemp,
  maxTemp,
  wireframe,
}: ThermalMaterialProps) {
  const materialRef = useRef<THREE.ShaderMaterial>(null);
  const uniforms = useMemo(
    () => ({
      uTemperature: { value: temperature },
      uMinTemp: { value: minTemp },
      uMaxTemp: { value: maxTemp },
    }),
    // Uniform objects are created once; values are updated imperatively below.
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [],
  );

  useEffect(() => {
    const m = materialRef.current;
    if (!m) return;
    m.uniforms.uTemperature.value = temperature;
    m.uniforms.uMinTemp.value = minTemp;
    m.uniforms.uMaxTemp.value = maxTemp;
  }, [temperature, minTemp, maxTemp]);

  return (
    <shaderMaterial
      ref={materialRef}
      uniforms={uniforms}
      vertexShader={THERMAL_VERTEX_SHADER}
      fragmentShader={THERMAL_FRAGMENT_SHADER}
      transparent
      side={THREE.DoubleSide}
      wireframe={wireframe}
    />
  );
}
