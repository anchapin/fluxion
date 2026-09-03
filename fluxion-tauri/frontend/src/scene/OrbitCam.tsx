import { useEffect, useRef } from "react";
import { useFrame, useThree } from "@react-three/fiber";
import { OrbitControls as OrbitControlsImpl } from "three/examples/jsm/controls/OrbitControls.js";
import * as THREE from "three";

interface OrbitCamProps {
  bounds: { min: [number, number, number]; max: [number, number, number] };
  onControlsReady?: (controls: OrbitControlsImpl | null) => void;
}

/**
 * Orbit controls with damping plus automatic camera fitting to the building
 * bounds (mirrors the preserved viewers' `fitCameraToScene`). Exposes the
 * controls instance so the ControlBar "Reset View" button can call
 * `controls.reset()` to return to the fitted framing.
 */
export function OrbitCam({ bounds, onControlsReady }: OrbitCamProps) {
  const camera = useThree((s) => s.camera);
  const gl = useThree((s) => s.gl);
  const controlsRef = useRef<OrbitControlsImpl | null>(null);

  useEffect(() => {
    const controls = new OrbitControlsImpl(camera, gl.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controlsRef.current = controls;
    onControlsReady?.(controls);
    return () => {
      controls.dispose();
      controlsRef.current = null;
      onControlsReady?.(null);
    };
  }, [camera, gl, onControlsReady]);

  useEffect(() => {
    const controls = controlsRef.current;
    if (!controls) return;

    const size = new THREE.Vector3(
      bounds.max[0] - bounds.min[0],
      bounds.max[1] - bounds.min[1],
      bounds.max[2] - bounds.min[2],
    );
    const center = new THREE.Vector3(
      (bounds.min[0] + bounds.max[0]) / 2,
      (bounds.min[1] + bounds.max[1]) / 2,
      (bounds.min[2] + bounds.max[2]) / 2,
    );
    const maxDim = Math.max(size.x, size.y, size.z, 1);
    const distance = maxDim * 2;
    camera.position.set(
      center.x + distance * 0.7,
      center.y + distance * 0.55,
      center.z + distance * 0.7,
    );
    controls.target.copy(center);
    controls.update();
    controls.saveState();
  }, [bounds, camera]);

  useFrame(() => {
    controlsRef.current?.update();
  });

  return null;
}

export type { OrbitControlsImpl };
