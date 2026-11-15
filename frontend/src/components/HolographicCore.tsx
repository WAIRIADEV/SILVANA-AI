
import { useEffect, useRef } from 'react'
import * as THREE from 'three'
import { OrbitControls } from 'three/addons/controls/OrbitControls.js'
import type { AssistantState } from '../types/assistant'
import { ASSISTANT_STATES } from '../types/assistant'
import './HolographicCore.css'

interface HolographicCoreProps {
  state: AssistantState
}

function HolographicCore({ state }: HolographicCoreProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const stateRef = useRef(state)

  useEffect(() => {
    stateRef.current = state
  }, [state])

  useEffect(() => {
    const containerElement = containerRef.current
    if (!containerElement) return

    const container = containerElement

    let disposed = false
    let frameId = 0
    let resizeObserver: ResizeObserver | null = null
    let mixer: THREE.AnimationMixer | null = null

    const scene = new THREE.Scene()

    const camera = new THREE.PerspectiveCamera(40, 1, 0.1, 100)
    camera.position.set(0, 0, 5.5)

    let renderer: THREE.WebGLRenderer

    try {
      renderer = new THREE.WebGLRenderer({
        alpha: true,
        antialias: false,
        powerPreference: 'low-power',
      })
    } catch (error) {
      console.warn('SILVANA: WebGL renderer unavailable.', error)
      return
    }

    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.25))

    renderer.setSize(
      Math.max(container.clientWidth, 1),
      Math.max(container.clientHeight, 1),
    )

    renderer.setClearColor(0x000000, 0)
    renderer.outputColorSpace = THREE.SRGBColorSpace

    const canvas = renderer.domElement
    canvas.style.cursor = 'grab'
    canvas.style.touchAction = 'none'

    container.appendChild(canvas)

    const controls = new OrbitControls(camera, canvas)

    controls.target.set(0, 0, 0)
    controls.enableDamping = true
    controls.dampingFactor = 0.06
    controls.enableZoom = true
    controls.zoomSpeed = 0.8
    controls.minDistance = 4.1
    controls.maxDistance = 8
    controls.enableRotate = true
    controls.rotateSpeed = 0.65
    controls.enablePan = false
    controls.minPolarAngle = Math.PI * 0.22
    controls.maxPolarAngle = Math.PI * 0.78
    controls.autoRotate = true
    controls.autoRotateSpeed = 0.4

    const renderScene = () => {
      if (!disposed) {
        renderer.render(scene, camera)
      }
    }

    controls.addEventListener('change', renderScene)

    const hologram = new THREE.Group()
    hologram.position.y = -0.05
    scene.add(hologram)

    const ownedMaterials = new Set<THREE.Material>()
    const ownedGeometries = new Set<THREE.BufferGeometry>()

    function disposeObject(root: THREE.Object3D) {
      root.traverse((object) => {
        const mesh = object as THREE.Mesh

        if (mesh.geometry) {
          ownedGeometries.add(mesh.geometry)
        }

        if (mesh.material) {
          const materials = Array.isArray(mesh.material)
            ? mesh.material
            : [mesh.material]

          materials.forEach((material) => {
            ownedMaterials.add(material)
          })
        }
      })
    }

    // STATE-DRIVEN ORBITAL ENERGY SYSTEM

    const orbitalGroup = new THREE.Group()
    scene.add(orbitalGroup)

    const waistOrbit = new THREE.Group()
    const upperOrbit = new THREE.Group()

    orbitalGroup.add(waistOrbit, upperOrbit)

    const waistMaterial = new THREE.MeshBasicMaterial({
      color: 0x22d3ee,
      transparent: true,
      opacity: 0.58,
      depthWrite: false,
    })

    const upperMaterial = new THREE.MeshBasicMaterial({
      color: 0x22d3ee,
      transparent: true,
      opacity: 0.5,
      depthWrite: false,
    })

    ownedMaterials.add(waistMaterial)
    ownedMaterials.add(upperMaterial)

    const waistRing = new THREE.Mesh(
      new THREE.TorusGeometry(1.05, 0.008, 6, 80),
      waistMaterial,
    )

    waistRing.position.y = -0.1
    waistRing.rotation.x = Math.PI / 2
    waistOrbit.add(waistRing)

    const upperRing = new THREE.Mesh(
      new THREE.TorusGeometry(0.78, 0.006, 6, 80),
      upperMaterial,
    )

    upperRing.position.y = 0.8
    upperRing.rotation.x = Math.PI / 2.4
    upperOrbit.add(upperRing)

    // FLOATING PARTICLES

    const particleCount = 100
    const positions = new Float32Array(particleCount * 3)

    for (let i = 0; i < particleCount; i++) {
      const angle = Math.random() * Math.PI * 2
      const radius = 1.2 + Math.random() * 1.1

      positions[i * 3] = Math.cos(angle) * radius
      positions[i * 3 + 1] = (Math.random() - 0.5) * 3.8
      positions[i * 3 + 2] = (Math.random() - 0.5) * 1.2
    }

    const particleGeometry = new THREE.BufferGeometry()

    particleGeometry.setAttribute(
      'position',
      new THREE.BufferAttribute(positions, 3),
    )

    ownedGeometries.add(particleGeometry)

    const particleMaterial = new THREE.PointsMaterial({
      color: 0x22d3ee,
      size: 0.018,
      transparent: true,
      opacity: 0.65,
      sizeAttenuation: true,
      depthWrite: false,
    })

    ownedMaterials.add(particleMaterial)

    const particles = new THREE.Points(
      particleGeometry,
      particleMaterial,
    )

    scene.add(particles)

    // HOLOGRAPHIC HUMANOID

    const modelPivot = new THREE.Group()
    hologram.add(modelPivot)

    const hologramMaterial = new THREE.MeshBasicMaterial({
      color: 0x22d3ee,
      wireframe: true,
      transparent: true,
      opacity: 0.78,
      side: THREE.DoubleSide,
      depthWrite: false,
    })

    ownedMaterials.add(hologramMaterial)

    async function loadHumanoid() {
      try {
        const { GLTFLoader } = await import(
          'three/addons/loaders/GLTFLoader.js'
        )

        if (disposed) return

        const loader = new GLTFLoader()

        const gltf = await loader.loadAsync(
          '/models/silvana-humanoid.glb',
        )

        if (disposed) {
          disposeObject(gltf.scene)
          return
        }

        const model = gltf.scene

        model.traverse((object) => {
          if (!(object instanceof THREE.Mesh)) return

          object.material = hologramMaterial
          object.frustumCulled = true
          object.renderOrder = 1

          if (object.geometry) {
            ownedGeometries.add(object.geometry)
          }
        })

        const bounds = new THREE.Box3().setFromObject(model)
        const size = bounds.getSize(new THREE.Vector3())
        const center = bounds.getCenter(new THREE.Vector3())

        if (size.y <= 0 || !Number.isFinite(size.y)) {
          throw new Error('The GLB model has invalid dimensions.')
        }

        const targetHeight = 3.25
        const scale = targetHeight / size.y

        model.scale.setScalar(scale)

        model.position.set(
          -center.x * scale,
          -center.y * scale,
          -center.z * scale,
        )

        modelPivot.add(model)

        if (gltf.animations.length > 0) {
          mixer = new THREE.AnimationMixer(model)

          const idleClip =
            gltf.animations.find((clip) =>
              /idle|rest|breath|stand/i.test(clip.name),
            ) ?? gltf.animations[0]

          mixer.clipAction(idleClip).play()

          console.info(
            `SILVANA: Model animation started: ${idleClip.name}`,
          )
        } else {
          console.info(
            'SILVANA: No skeletal animation found. Using procedural idle motion.',
          )
        }

        renderScene()

        console.info('SILVANA: Female holographic model loaded.')
      } catch (error) {
        if (!disposed) {
          console.error(
            'SILVANA: Failed to load female GLB model.',
            error,
          )
        }
      }
    }

    // RESPONSIVE VIEWPORT

    function resize() {
      const width = container.clientWidth
      const height = container.clientHeight

      if (!width || !height) return

      camera.aspect = width / height
      camera.updateProjectionMatrix()

      renderer.setSize(width, height)

      renderScene()
    }

    resizeObserver = new ResizeObserver(resize)
    resizeObserver.observe(container)

    // STATE TRANSITION TRACKING

    const activeColor = new THREE.Color()
    const targetColor = new THREE.Color()

    let activeState: AssistantState = stateRef.current

    function applyAssistantState(nextState: AssistantState) {
      if (nextState === activeState) return

      activeState = nextState

      const config = ASSISTANT_STATES[nextState]

      targetColor.setHex(config.color)

      console.info(`SILVANA: Assistant state changed to ${nextState}.`)
    }

    targetColor.setHex(ASSISTANT_STATES[activeState].color)
    activeColor.copy(targetColor)

    // ANIMATION LOOP

    const timer = new THREE.Timer()

    let elapsed = 0

    function animate() {
      if (disposed || document.hidden) {
        frameId = 0
        return
      }

      frameId = requestAnimationFrame(animate)

      timer.update()

      const delta = Math.min(timer.getDelta(), 0.05)

      elapsed += delta

      applyAssistantState(stateRef.current)

      // Smooth colour transitions instead of abrupt colour changes.
      activeColor.lerp(targetColor, 1 - Math.exp(-delta * 3.5))

      hologramMaterial.color.copy(activeColor)
      waistMaterial.color.copy(activeColor)
      upperMaterial.color.copy(activeColor)
      particleMaterial.color.copy(activeColor)

      // State-dependent intensity.
      const pulseSpeed =
        activeState === 'thinking' ? 2.8 :
        activeState === 'executing' ? 4.2 :
        activeState === 'verifying' ? 1.8 :
        activeState === 'error' ? 3.5 : 1.1

      const pulseStrength =
        activeState === 'idle' ? 0.009 :
        activeState === 'completed' ? 0.014 : 0.025

      const breathing =
        1 + Math.sin(elapsed * pulseSpeed) * pulseStrength

      modelPivot.scale.setScalar(breathing)

      // NATURAL MODEL SWAY
      modelPivot.rotation.z =
        Math.sin(elapsed * 0.65) * 0.012

      modelPivot.rotation.x =
        Math.sin(elapsed * 0.45) * 0.008

      // ORBITAL ENERGY SYSTEM
      const orbitSpeed =
        activeState === 'thinking' ? 1.2 :
        activeState === 'executing' ? 1.8 :
        activeState === 'verifying' ? 0.45 : 0.65

      waistOrbit.rotation.y += delta * orbitSpeed

      waistOrbit.rotation.z =
        Math.sin(elapsed * 0.7) * 0.2

      upperOrbit.rotation.y -= delta * orbitSpeed * 1.5

      upperOrbit.rotation.x =
        Math.sin(elapsed * 0.9) * 0.25

      upperOrbit.rotation.z += delta * 0.25

      // PARTICLES
      particles.rotation.y += delta * (
        activeState === 'thinking' ? 0.055 : 0.025
      )

      particles.rotation.x =
        Math.sin(elapsed * 0.3) * 0.04

      mixer?.update(delta)

      controls.update()

      renderer.render(scene, camera)
    }

    function startAnimation() {
      if (
        disposed ||
        document.hidden ||
        frameId !== 0
      ) {
        return
      }

      timer.reset()

      frameId = requestAnimationFrame(animate)
    }

    function stopAnimation() {
      if (frameId !== 0) {
        cancelAnimationFrame(frameId)
        frameId = 0
      }
    }

    function handleVisibilityChange() {
      if (document.hidden) {
        stopAnimation()
      } else {
        startAnimation()
        renderScene()
      }
    }

    document.addEventListener(
      'visibilitychange',
      handleVisibilityChange,
    )

    void loadHumanoid()

    startAnimation()

    // RESOURCE CLEANUP

    return () => {
      disposed = true

      stopAnimation()

      document.removeEventListener(
        'visibilitychange',
        handleVisibilityChange,
      )

      controls.removeEventListener('change', renderScene)
      controls.dispose()

      mixer?.stopAllAction()
      mixer?.uncacheRoot(modelPivot)

      resizeObserver?.disconnect()

      disposeObject(scene)

      ownedGeometries.forEach((geometry) => geometry.dispose())
      ownedMaterials.forEach((material) => material.dispose())

      renderer.dispose()
      renderer.domElement.remove()
    }
  }, [])

  return (
    <div
      ref={containerRef}
      className="holographic-canvas"
      role="img"
      aria-label="SILVANA female holographic intelligence avatar"
    />
  )
}

export default HolographicCore
