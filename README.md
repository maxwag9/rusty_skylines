# Rusty Skylines
Cities: Skylines 2 but better!

The goal is to make rusty skylines: 
A Cities: Skylines inspired game featuring better features and way better performance as a priority.

<img width="2562" height="1430" alt="Roads" src="https://github.com/user-attachments/assets/11597c41-9dca-4a05-aea6-fb61fac35a9e" />

---
## Priorities:
- **Small game size** (under 50 MB TOTAL, currently 7MB)  🥵 Compared to Cities Skylines 2 at 60000MB  

- Procedural textures and models  

- Extremely performant and realistic traffic Sim (very smart cars)

- Extreme, **infinite** scale. Trucks can deliver from one country to the other.

- **GRIDLESS** procedural building  

- Great Graphics, running at target: RTX 4060, 2560x1440, 100FPS  

- Overall, multithreaded highly parallel simulation for nearly everything, with deferred, long tick rates per chunk  

- Modding in the future with lua for low-performance and wasm for higher complexity/performance  

---
## It currently features:  

A 3D scene with orbiting mechanics and movement of the camera.  

The 3D scene has:  

- **Infinite, chunked, procedural Terrain**  

- Sun and moon with moon phases and correct orbiting, both dependent on the sun.  

- The sun oscillates with the orbital tilt of the earth, set at Munich.  

- **Stars rendered from a real life database (Hipparcos Catalogue) of 118k stars**  

- Water rendered with Moon and Sun reflections  

- Roads and cars.

---
## More features:
**Procedural Texture Manager** inside my entire RenderManager (see my wgpu_render_manager repository)  

**GTAO Ambient Occlusion**  

**Vignette, Fog, Tonemapping**  

**Cascaded Shadows (4 Cascades)**  

Driving one car with WASD.

A WIP GUI Editor with the ability to edit any GUI in-game and save it,
allowing expressions and logic and animations to be available through modding. 

Hot-reloadable shaders (and by extension texture shaders!)

I won't hard-code any GUI, I'll just use the same thing I provide as modding, 
therefore creating great modding support. Win-Win!

All graphics are MSAA anti-aliased. 
(I didn't see a performance hit yet, but we'll see about that!)

## Terrain

<img width="2562" height="1430" alt="Screenshot_20260123_131033" src="https://github.com/user-attachments/assets/403e6fcd-961e-4ada-8f93-b5c556119229" />

## Cars

<img width="2562" height="1430" alt="Screenshot_20260211_174818" src="https://github.com/user-attachments/assets/0ce379bf-612e-40c4-9ef6-0598a8d2da80" />

## Intersections

<img width="2562" height="1430" alt="Screenshot_20260122_013021" src="https://github.com/user-attachments/assets/0a00a2cb-b2e8-4e5b-819a-c8c487e615b4" />

Rusty Skylines is open for modding and non-commercial use.
Attribution required, no resale or commercial distribution of the base game.
Donations allowed as per license terms.
