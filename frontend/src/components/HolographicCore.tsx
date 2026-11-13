
import { useEffect, useRef } from 'react'
import * as THREE from 'three'
import { OrbitControls } from 'three/addons/controls/OrbitControls.js'
import './HolographicCore.css'

function HolographicCore() {
  const containerRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const containerElement = containerRef.current
    if (!containerElement) return

    const container = containerElement

    // Track component lifecycle and animation resources.
    let disposed = false
    let frameId = 0
    let resizeObserver: ResizeObserver | null = null
    let mixer: THREE.AnimationMixer | null = null

    // --------------------------------------------------
    // 1. SCENE AND CAMERA
    // --------------------------------------------------

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

    // Limit rendering resolution to protect lower-powered hardware.
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

    // --------------------------------------------------
    // 2. CAMERA CONTROLS
    // --------------------------------------------------

    const controls = new OrbitControls(camera, canvas)

    controls.target.set(0, 0, 0)

    controls.enableDamping = true
    controls.dampingFactor = 0.06

    // Mouse wheel zoom.
    controls.enableZoom = true
    controls.zoomSpeed = 0.8
    controls.minDistance = 4.1
    controls.maxDistance = 8

    // Mouse drag rotation.
    controls.enableRotate = true
    controls.rotateSpeed = 0.65

    // Prevent accidental camera panning.
    controls.enablePan = false

    // Restrict vertical camera movement.
    controls.minPolarAngle = Math.PI * 0.22
    controls.maxPolarAngle = Math.PI * 0.78

    // Enable automatic camera rotation.
    // This is intentionally independent of the OS reduced-motion setting.
    controls.autoRotate = true
    controls.autoRotateSpeed = 0.4

    const renderScene = () => {
      if (!disposed) {
        renderer.render(scene, camera)
      }
    }

    controls.addEventListener('change', renderScene)

    // --------------------------------------------------
    // 3. HOLOGRAM CONTAINER
    // --------------------------------------------------

    const hologram = new THREE.Group()
    hologram.position.y = -0.05

    scene.add(hologram)

    // Track resources to ensure everything is disposed properly.
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

    // --------------------------------------------------
    // 4. INDEPENDENT ORBITAL ENERGY SYSTEM
    // --------------------------------------------------

    const orbitalGroup = new THREE.Group()
    scene.add(orbitalGroup)

    // Separate parent groups allow each ring to rotate independently.
    const waistOrbit = new THREE.Group()
    const upperOrbit = new THREE.Group()

    orbitalGroup.add(waistOrbit, upperOrbit)

    // Purple energy ring.
    const waistMaterial = new THREE.MeshBasicMaterial({
      color: 0x8c63ff,
      transparent: true,
      opacity: 0.58,
      depthWrite: false,
    })

    // Cyan energy ring.
    const upperMaterial = new THREE.MeshBasicMaterial({
      color: 0x42dfff,
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

    // --------------------------------------------------
    // 5. FLOATING PARTICLES
    // --------------------------------------------------

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
      color: 0x65dcff,
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

    // --------------------------------------------------
    // 6. FEMALE HOLOGRAPHIC MODEL
    // --------------------------------------------------

    const modelPivot = new THREE.Group()
    hologram.add(modelPivot)

    const hologramMaterial = new THREE.MeshBasicMaterial({
      color: 0x42dfff,
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

        // Apply the holographic material to the entire model.
        model.traverse((object) => {
          if (!(object instanceof THREE.Mesh)) return

          object.material = hologramMaterial
          object.frustumCulled = true
          object.renderOrder = 1

          if (object.geometry) {
            ownedGeometries.add(object.geometry)
          }
        })

        // Calculate model dimensions and center it.
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

        // If the GLB contains animation clips, use an idle-compatible clip.
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

    // --------------------------------------------------
    // 7. RESPONSIVE VIEWPORT
    // --------------------------------------------------

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

    // --------------------------------------------------
    // 8. IDLE AND ORBITAL ANIMATION
    // --------------------------------------------------

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

      // Ambient animations intentionally remain enabled.
      // They are not disabled by the operating system preference.

      // GENTLE HOLOGRAM FLOATING
      hologram.position.y =
        -0.05 + Math.sin(elapsed * 0.85) * 0.035

      // SUBTLE BREATHING EFFECT
      const breathing = 1 + Math.sin(elapsed * 1.1) * 0.009

      modelPivot.scale.setScalar(breathing)

      // NATURAL MODEL SWAY
      modelPivot.rotation.z =
        Math.sin(elapsed * 0.65) * 0.012

      modelPivot.rotation.x =
        Math.sin(elapsed * 0.45) * 0.008

      // PURPLE WAIST RING
      // Steady clockwise rotation with a gentle orbital tilt.
      waistOrbit.rotation.y += delta * 0.65

      waistOrbit.rotation.z =
        Math.sin(elapsed * 0.7) * 0.2

      // CYAN UPPER RING
      // Faster counter-clockwise rotation on an independent axis.
      upperOrbit.rotation.y -= delta * 1.0

      upperOrbit.rotation.x =
        Math.sin(elapsed * 0.9) * 0.25

      upperOrbit.rotation.z += delta * 0.25

      // AMBIENT PARTICLE MOVEMENT
      particles.rotation.y += delta * 0.025

      particles.rotation.x =
        Math.sin(elapsed * 0.3) * 0.04

      // PLAY SKELETAL ANIMATION IF AVAILABLE.
      mixer?.update(delta)

      // Update camera damping and automatic rotation.
      controls.update()

      renderer.render(scene, camera)
    }

    // --------------------------------------------------
    // 9. ANIMATION LIFECYCLE
    // --------------------------------------------------

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

    // Pause rendering when the browser tab is hidden.
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

    // --------------------------------------------------
    // 10. RESOURCE CLEANUP
    // --------------------------------------------------

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
