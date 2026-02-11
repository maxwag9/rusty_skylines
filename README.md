# rusty_skylines
Cities: Skylines 2 but better!

The goal is to make rusty skylines: 
A Cities: Skylines inspired game featuring better features and way better performance as a priority.

Modding is also a priority!

It currently features:

A 3D scene with orbiting mechanics and movement of the camera.
The 3D scene has:
Terrain with shadows
Sun and moon with moon phases and correct orbiting, both dependent on the sun. The sun oscillates with the orbital tilt of the earth, set at Munich.
Stars rendered from a real life database (Hipparcos Catalogue) of 118k stars
Water rendered with Moon and Sun reflections
Roads and cars.

More features:
Procedural Texture Manager inside my entire RenderManager (see my wgpu_render_manager repository)
GTAO Ambient Occlusion
Vignette, Fog, Tonemapping
Driving a car with WASD.

A WIP GUI Editor with the ability to edit any GUI in-game and save it,
allowing expressions and logic and animations to be available through modding. 

Hot-reloadable shaders (and by extension texture shaders!)

I won't hard-code any GUI, I'll just use the same thing I provide as modding, 
therefore creating great modding support. Win-Win!

All graphics are MSAA anti-aliased. 
(I didn't see a performance hit yet, but we'll see about that!)

<img width="3442" height="1442" alt="Screenshot_20251221_161420" src="https://github.com/user-attachments/assets/bf61deae-10b0-4bdf-b9a7-3418d378baaf" />

<img width="2562" height="1431" alt="Screenshot_20251125_115518" src="https://github.com/user-attachments/assets/6c02bf24-2ad1-44ef-b57f-bb6559dbb6d1" />

Rusty Skylines is open for modding and non-commercial use.
Attribution required, no resale or commercial distribution of the base game.
Donations allowed as per license terms.
