'use client';

import { useRef, useState, useMemo, useEffect } from 'react';
import { Canvas, useThree } from '@react-three/fiber';
import { OrbitControls, Line, Html } from '@react-three/drei';
import * as THREE from 'three';

// Element colors - distinct, vibrant colors matching Materials Project style
const ELEMENT_COLORS: Record<string, string> = {
  // Common non-metals - bright distinct colors
  H: '#FFFFFF',
  C: '#505050',
  N: '#3050F8',
  O: '#FF0000',  // Bright red for oxygen
  F: '#90E050',
  S: '#FFFF30',
  P: '#FF8000',
  Cl: '#1FF01F',
  Br: '#A62929',
  I: '#940094',
  // Alkali & Alkaline Earth
  Li: '#CC80FF',
  Na: '#AB5CF2',
  K: '#8F40D4',
  Mg: '#8AFF00',
  Ca: '#3DFF00',
  Sr: '#00FF00',
  Ba: '#00C900',
  // Transition metals - earthy/metallic tones
  Ti: '#BFC2C7',
  V: '#A6A6AB',
  Cr: '#8A99C7',
  Mn: '#9C7AC7',
  Fe: '#B87333',  // Copper-brown for iron (distinct from red oxygen)
  Co: '#F090A0',
  Ni: '#50D050',
  Cu: '#C88033',
  Zn: '#7D80B0',
  Y: '#94FFFF',
  Zr: '#94E0E0',
  Nb: '#73C2C9',
  Mo: '#54B5B5',
  Ru: '#248F8F',
  Rh: '#0A7D8C',
  Pd: '#006985',
  Ag: '#C0C0C0',
  Cd: '#FFD98F',
  Hf: '#4DC2FF',
  Ta: '#4DA6FF',
  W: '#2194D6',
  Re: '#267DAB',
  Os: '#266696',
  Ir: '#175487',
  Pt: '#D0D0E0',
  Au: '#FFD123',
  // Post-transition metals
  Al: '#BFA6A6',
  Ga: '#C28F8F',
  In: '#A67573',
  Sn: '#668080',
  Pb: '#575961',
  Bi: '#9E4FB5',
  // Metalloids
  Si: '#F0C8A0',
  Ge: '#668F8F',
  As: '#BD80E3',
  Se: '#FFA100',
  Sb: '#9E63B5',
  Te: '#D47A00',
  // Lanthanides
  La: '#70D4FF',
  Ce: '#FFFFC7',
  Pr: '#D9FFC7',
  Nd: '#C7FFC7',
  Sm: '#8FFFC7',
  Eu: '#61FFC7',
  Gd: '#45FFC7',
  Tb: '#30FFC7',
  Dy: '#1FFFC7',
  Ho: '#00FF9C',
  Er: '#00E675',
  Tm: '#00D452',
  Yb: '#00BF38',
  Lu: '#00AB24',
  // Actinides
  U: '#008FFF',
  Np: '#0080FF',
  Pu: '#006BFF',
  Rb: '#702EB0',
};

// Element radii (covalent radii in Angstroms for visualization)
const ELEMENT_RADII: Record<string, number> = {
  H: 0.31, C: 0.76, N: 0.71, O: 0.66, F: 0.57, S: 1.05, P: 1.07, Cl: 1.02,
  Li: 1.28, Na: 1.66, K: 2.03, Mg: 1.41, Ca: 1.76, Al: 1.21, Si: 1.11,
  Ti: 1.60, V: 1.53, Cr: 1.39, Mn: 1.39, Fe: 1.32, Co: 1.26, Ni: 1.24,
  Cu: 1.32, Zn: 1.22, Ga: 1.22, Ge: 1.20, As: 1.19, Se: 1.20, Br: 1.20,
  Rb: 2.20, Sr: 1.95, Y: 1.90, Zr: 1.75, Nb: 1.64, Mo: 1.54, Ru: 1.46,
  Rh: 1.42, Pd: 1.39, Ag: 1.45, Cd: 1.44, In: 1.42, Sn: 1.39, Sb: 1.39,
  Te: 1.38, I: 1.39, Ba: 2.15, La: 2.07, Ce: 2.04, Pr: 2.03, Nd: 2.01,
  Sm: 1.98, Eu: 1.98, Gd: 1.96, Tb: 1.94, Dy: 1.92, Ho: 1.92, Er: 1.89,
  Tm: 1.90, Yb: 1.87, Lu: 1.87, Hf: 1.75, Ta: 1.70, W: 1.62, Re: 1.51,
  Os: 1.44, Ir: 1.41, Pt: 1.36, Au: 1.36, Pb: 1.46, Bi: 1.48, U: 1.96,
};

// Bond distance threshold multiplier (sum of covalent radii * this factor)
const BOND_TOLERANCE = 1.3;

interface Site {
  species: Array<{ element: string; occu: number }>;
  xyz: number[];
  label: string;
  abc: number[];
  properties?: { magmom?: number };
}

interface Structure {
  lattice: {
    matrix: number[][];
    a: number;
    b: number;
    c: number;
    alpha: number;
    beta: number;
    gamma: number;
    volume: number;
  };
  sites: Site[];
}

interface CrystalViewerProps {
  structure: Structure | null;
  isLoading?: boolean;
}

// Atom component - a sphere representing an atom
function Atom({
  position,
  element,
  radius,
  onHover,
  onUnhover,
  isHovered,
  index,
}: {
  position: [number, number, number];
  element: string;
  radius: number;
  onHover: (info: { element: string; position: number[]; index: number }) => void;
  onUnhover: () => void;
  isHovered: boolean;
  index: number;
}) {
  const meshRef = useRef<THREE.Mesh>(null);
  const color = ELEMENT_COLORS[element] || '#808080';

  return (
    <mesh
      ref={meshRef}
      position={position}
      onPointerOver={(e) => {
        e.stopPropagation();
        onHover({ element, position: [...position], index });
      }}
      onPointerOut={onUnhover}
    >
      <sphereGeometry args={[radius, 32, 32]} />
      <meshStandardMaterial
        color={color}
        roughness={0.2}
        metalness={0.1}
        emissive={isHovered ? color : '#000000'}
        emissiveIntensity={isHovered ? 0.4 : 0}
      />
    </mesh>
  );
}

// Bond component - a cylinder connecting two atoms
function Bond({
  start,
  end,
  color = '#666666',
  radius = 0.08,
}: {
  start: [number, number, number];
  end: [number, number, number];
  color?: string;
  radius?: number;
}) {
  const ref = useRef<THREE.Mesh>(null);

  // Calculate bond geometry
  const { position, rotation, length } = useMemo(() => {
    const startVec = new THREE.Vector3(...start);
    const endVec = new THREE.Vector3(...end);
    
    // Midpoint for position
    const midpoint = new THREE.Vector3().addVectors(startVec, endVec).multiplyScalar(0.5);
    
    // Length of bond
    const bondLength = startVec.distanceTo(endVec);
    
    // Direction and rotation
    const direction = new THREE.Vector3().subVectors(endVec, startVec).normalize();
    const quaternion = new THREE.Quaternion();
    quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), direction);
    const euler = new THREE.Euler().setFromQuaternion(quaternion);
    
    return {
      position: [midpoint.x, midpoint.y, midpoint.z] as [number, number, number],
      rotation: [euler.x, euler.y, euler.z] as [number, number, number],
      length: bondLength,
    };
  }, [start, end]);

  return (
    <mesh ref={ref} position={position} rotation={rotation}>
      <cylinderGeometry args={[radius, radius, length, 16]} />
      <meshStandardMaterial color={color} roughness={0.4} metalness={0.2} />
    </mesh>
  );
}

// Unit cell wireframe
function UnitCell({ matrix, offset }: { matrix: number[][]; offset: [number, number, number] }) {
  const [a, b, c] = matrix;
  const [ox, oy, oz] = offset;
  
  // Calculate the 8 vertices of the unit cell (shifted by offset)
  const origin: [number, number, number] = [ox, oy, oz];
  const v1: [number, number, number] = [a[0] + ox, a[1] + oy, a[2] + oz];
  const v2: [number, number, number] = [b[0] + ox, b[1] + oy, b[2] + oz];
  const v3: [number, number, number] = [c[0] + ox, c[1] + oy, c[2] + oz];
  const v12: [number, number, number] = [a[0] + b[0] + ox, a[1] + b[1] + oy, a[2] + b[2] + oz];
  const v13: [number, number, number] = [a[0] + c[0] + ox, a[1] + c[1] + oy, a[2] + c[2] + oz];
  const v23: [number, number, number] = [b[0] + c[0] + ox, b[1] + c[1] + oy, b[2] + c[2] + oz];
  const v123: [number, number, number] = [a[0] + b[0] + c[0] + ox, a[1] + b[1] + c[1] + oy, a[2] + b[2] + c[2] + oz];

  // Define the 12 edges of the unit cell
  const edges: Array<[[number, number, number], [number, number, number]]> = [
    // Bottom face
    [origin, v1],
    [origin, v2],
    [v1, v12],
    [v2, v12],
    // Top face
    [v3, v13],
    [v3, v23],
    [v13, v123],
    [v23, v123],
    // Vertical edges
    [origin, v3],
    [v1, v13],
    [v2, v23],
    [v12, v123],
  ];

  return (
    <group>
      {edges.map((edge, i) => (
        <Line
          key={i}
          points={edge}
          color="#374151"
          lineWidth={1.5}
        />
      ))}
    </group>
  );
}

// Calculate bonds between atoms based on distance
function calculateBonds(
  atoms: Array<{ position: [number, number, number]; element: string; index: number }>
): Array<{ start: [number, number, number]; end: [number, number, number]; color: string }> {
  const bonds: Array<{ start: [number, number, number]; end: [number, number, number]; color: string }> = [];
  
  for (let i = 0; i < atoms.length; i++) {
    for (let j = i + 1; j < atoms.length; j++) {
      const atom1 = atoms[i];
      const atom2 = atoms[j];
      
      // Calculate distance
      const dx = atom1.position[0] - atom2.position[0];
      const dy = atom1.position[1] - atom2.position[1];
      const dz = atom1.position[2] - atom2.position[2];
      const distance = Math.sqrt(dx * dx + dy * dy + dz * dz);
      
      // Get covalent radii
      const r1 = ELEMENT_RADII[atom1.element] || 1.5;
      const r2 = ELEMENT_RADII[atom2.element] || 1.5;
      
      // Check if bonded (distance less than sum of covalent radii * tolerance)
      const maxBondDistance = (r1 + r2) * BOND_TOLERANCE;
      
      if (distance < maxBondDistance && distance > 0.5) {
        // Use a neutral color for bonds
        bonds.push({
          start: atom1.position,
          end: atom2.position,
          color: '#888888',
        });
      }
    }
  }
  
  return bonds;
}

// Main scene component
function CrystalScene({ structure }: { structure: Structure }) {
  const [hoveredAtom, setHoveredAtom] = useState<{
    element: string;
    position: number[];
    index: number;
  } | null>(null);

  // Calculate center of structure for camera focus
  const { center, atoms, bonds, unitCellOffset } = useMemo(() => {
    const sites = structure.sites;
    const positions = sites.map((s) => s.xyz);
    
    // Calculate bounding box
    const minX = Math.min(...positions.map((p) => p[0]));
    const maxX = Math.max(...positions.map((p) => p[0]));
    const minY = Math.min(...positions.map((p) => p[1]));
    const maxY = Math.max(...positions.map((p) => p[1]));
    const minZ = Math.min(...positions.map((p) => p[2]));
    const maxZ = Math.max(...positions.map((p) => p[2]));

    const centerX = (minX + maxX) / 2;
    const centerY = (minY + maxY) / 2;
    const centerZ = (minZ + maxZ) / 2;

    const scaleFactor = 0.35; // Atom size multiplier

    const atomData = sites.map((site, index) => {
      const element = site.species[0]?.element || site.label;
      const baseRadius = ELEMENT_RADII[element] || 1.0;
      return {
        position: [
          site.xyz[0] - centerX,
          site.xyz[1] - centerY,
          site.xyz[2] - centerZ,
        ] as [number, number, number],
        element,
        radius: baseRadius * scaleFactor,
        index,
      };
    });

    // Calculate bonds
    const bondData = calculateBonds(atomData);

    // Calculate unit cell offset to center it
    const [a, b, c] = structure.lattice.matrix;
    const cellCenterX = (a[0] + b[0] + c[0]) / 2;
    const cellCenterY = (a[1] + b[1] + c[1]) / 2;
    const cellCenterZ = (a[2] + b[2] + c[2]) / 2;
    
    const offsetX = -centerX - cellCenterX + (a[0] + b[0] + c[0]) / 2;
    const offsetY = -centerY - cellCenterY + (a[1] + b[1] + c[1]) / 2;
    const offsetZ = -centerZ - cellCenterZ + (a[2] + b[2] + c[2]) / 2;

    return {
      center: [centerX, centerY, centerZ],
      atoms: atomData,
      bonds: bondData,
      unitCellOffset: [-centerX, -centerY, -centerZ] as [number, number, number],
    };
  }, [structure]);

  return (
    <>
      {/* Lighting */}
      <ambientLight intensity={0.7} />
      <directionalLight position={[10, 10, 10]} intensity={0.9} />
      <directionalLight position={[-10, -10, -5]} intensity={0.5} />
      <pointLight position={[0, 0, 15]} intensity={0.4} />

      {/* Unit cell */}
      <UnitCell matrix={structure.lattice.matrix} offset={unitCellOffset} />

      {/* Bonds */}
      {bonds.map((bond, i) => (
        <Bond
          key={`bond-${i}`}
          start={bond.start}
          end={bond.end}
          color={bond.color}
          radius={0.06}
        />
      ))}

      {/* Atoms */}
      {atoms.map((atom, i) => (
        <Atom
          key={i}
          position={atom.position}
          element={atom.element}
          radius={atom.radius}
          index={atom.index}
          onHover={setHoveredAtom}
          onUnhover={() => setHoveredAtom(null)}
          isHovered={hoveredAtom?.index === atom.index}
        />
      ))}

      {/* Hovered atom info tooltip */}
      {hoveredAtom && (
        <Html position={[hoveredAtom.position[0], hoveredAtom.position[1] + 1, hoveredAtom.position[2]]}>
          <div className="rounded-lg bg-zinc-900 px-3 py-2 text-sm text-white shadow-lg whitespace-nowrap">
            <span className="font-semibold">{hoveredAtom.element}</span>
            <span className="text-zinc-400 ml-2">
              ({hoveredAtom.position.map((v) => v.toFixed(3)).join(', ')})
            </span>
            <span className="text-zinc-500 ml-2">index:{hoveredAtom.index}</span>
          </div>
        </Html>
      )}

      {/* Controls */}
      <OrbitControls
        enablePan={true}
        enableZoom={true}
        enableRotate={true}
        autoRotate={false}
        minDistance={3}
        maxDistance={50}
      />
    </>
  );
}

// Legend component showing element colors
function ElementLegend({ structure }: { structure: Structure }) {
  const elements = useMemo(() => {
    const uniqueElements = new Set<string>();
    structure.sites.forEach((site) => {
      const element = site.species[0]?.element || site.label;
      uniqueElements.add(element);
    });
    return Array.from(uniqueElements);
  }, [structure]);

  return (
    <div className="absolute bottom-4 right-4 flex gap-2">
      {elements.map((element) => (
        <div
          key={element}
          className="flex items-center justify-center rounded-full px-3 py-1.5 text-sm font-bold shadow-md"
          style={{ 
            backgroundColor: ELEMENT_COLORS[element] || '#808080',
            color: ['O', 'Fe', 'C', 'N', 'S'].includes(element) ? '#ffffff' : '#000000',
          }}
        >
          {element}
        </div>
      ))}
    </div>
  );
}

// Fixed axes indicator overlay (HTML-based, stays in corner)
function AxesOverlay({ canvasRef }: { canvasRef: React.RefObject<HTMLDivElement | null> }) {
  const [rotation, setRotation] = useState({ x: 0, y: 0, z: 0 });
  
  useEffect(() => {
    // We'll update the axes based on the camera rotation via a custom event
    const handleCameraMove = (e: CustomEvent<{ x: number; y: number; z: number }>) => {
      setRotation(e.detail);
    };
    
    window.addEventListener('cameraRotation' as any, handleCameraMove as any);
    return () => {
      window.removeEventListener('cameraRotation' as any, handleCameraMove as any);
    };
  }, []);

  const axisLength = 25;
  const center = { x: 30, y: 70 }; // Center of the axes indicator

  // Apply rotation to axes (simplified rotation matrix)
  const rotatePoint = (x: number, y: number, z: number) => {
    const cosX = Math.cos(rotation.x);
    const sinX = Math.sin(rotation.x);
    const cosY = Math.cos(rotation.y);
    const sinY = Math.sin(rotation.y);
    
    // Rotate around Y then X (simplified)
    let newX = x * cosY + z * sinY;
    let newZ = -x * sinY + z * cosY;
    let newY = y * cosX - newZ * sinX;
    newZ = y * sinX + newZ * cosX;
    
    return { x: newX, y: -newY }; // Flip Y for screen coordinates
  };

  const xEnd = rotatePoint(axisLength, 0, 0);
  const yEnd = rotatePoint(0, axisLength, 0);
  const zEnd = rotatePoint(0, 0, axisLength);

  return (
    <div className="absolute bottom-4 left-4 pointer-events-none">
      <svg width="60" height="60" viewBox="0 0 60 100">
        {/* X axis - Red */}
        <line
          x1={center.x}
          y1={center.y}
          x2={center.x + xEnd.x}
          y2={center.y + xEnd.y}
          stroke="#ef4444"
          strokeWidth="2.5"
          strokeLinecap="round"
        />
        {/* Y axis - Green */}
        <line
          x1={center.x}
          y1={center.y}
          x2={center.x + yEnd.x}
          y2={center.y + yEnd.y}
          stroke="#22c55e"
          strokeWidth="2.5"
          strokeLinecap="round"
        />
        {/* Z axis - Blue */}
        <line
          x1={center.x}
          y1={center.y}
          x2={center.x + zEnd.x}
          y2={center.y + zEnd.y}
          stroke="#3b82f6"
          strokeWidth="2.5"
          strokeLinecap="round"
        />
      </svg>
    </div>
  );
}

// Camera rotation tracker
function CameraTracker() {
  const { camera } = useThree();
  
  useEffect(() => {
    const interval = setInterval(() => {
      const euler = new THREE.Euler().setFromQuaternion(camera.quaternion);
      window.dispatchEvent(
        new CustomEvent('cameraRotation', {
          detail: { x: euler.x, y: euler.y, z: euler.z },
        })
      );
    }, 50);
    
    return () => clearInterval(interval);
  }, [camera]);
  
  return null;
}

// Control buttons
function ViewerControls({
  onReset,
  onFullscreen,
}: {
  onReset: () => void;
  onFullscreen: () => void;
}) {
  return (
    <div className="absolute right-4 top-4 flex flex-col gap-2">
      <button
        onClick={onFullscreen}
        className="rounded-lg bg-white p-2 shadow-md transition-colors hover:bg-zinc-100"
        title="Fullscreen"
      >
        <svg className="h-5 w-5 text-zinc-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 8V4m0 0h4M4 4l5 5m11-1V4m0 0h-4m4 0l-5 5M4 16v4m0 0h4m-4 0l5-5m11 5l-5-5m5 5v-4m0 4h-4" />
        </svg>
      </button>
      <button
        onClick={onReset}
        className="rounded-lg bg-white p-2 shadow-md transition-colors hover:bg-zinc-100"
        title="Reset View"
      >
        <svg className="h-5 w-5 text-zinc-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
        </svg>
      </button>
    </div>
  );
}

// Main CrystalViewer component
export function CrystalViewer({ structure, isLoading = false }: CrystalViewerProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [key, setKey] = useState(0);

  const handleReset = () => {
    setKey((k) => k + 1);
  };

  const handleFullscreen = () => {
    if (containerRef.current) {
      if (document.fullscreenElement) {
        document.exitFullscreen();
      } else {
        containerRef.current.requestFullscreen();
      }
    }
  };

  if (isLoading) {
    return (
      <div className="flex h-full w-full items-center justify-center bg-white">
        <div className="text-center">
          <div className="mx-auto h-8 w-8 animate-spin rounded-full border-4 border-zinc-200 border-t-blue-500" />
          <p className="mt-2 text-sm text-zinc-500">Loading structure...</p>
        </div>
      </div>
    );
  }

  if (!structure || !structure.sites || structure.sites.length === 0) {
    return (
      <div className="flex h-full w-full items-center justify-center bg-zinc-50">
        <div className="p-8 text-center">
          <svg
            className="mx-auto h-16 w-16 text-zinc-300"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={1.5}
              d="M14 10l-2 1m0 0l-2-1m2 1v2.5M20 7l-2 1m2-1l-2-1m2 1v2.5M14 4l-2-1-2 1M4 7l2-1M4 7l2 1M4 7v2.5M12 21l-2-1m2 1l2-1m-2 1v-2.5M6 18l-2-1v-2.5M18 18l2-1v-2.5"
            />
          </svg>
          <p className="mt-4 text-lg font-medium text-zinc-500">
            No Structure Data
          </p>
          <p className="mt-2 text-sm text-zinc-400">
            Structure information not available for this material
          </p>
        </div>
      </div>
    );
  }

  return (
    <div ref={containerRef} className="relative h-full w-full bg-white">
      <Canvas
        key={key}
        camera={{ position: [12, 12, 12], fov: 50 }}
        style={{ background: '#ffffff' }}
      >
        <CrystalScene structure={structure} />
        <CameraTracker />
      </Canvas>
      <ViewerControls onReset={handleReset} onFullscreen={handleFullscreen} />
      <AxesOverlay canvasRef={containerRef} />
      <ElementLegend structure={structure} />
    </div>
  );
}

export default CrystalViewer;
