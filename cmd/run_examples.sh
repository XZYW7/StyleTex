image_path="load/style_images/woman_style_of_Milton_Glaser.png"
image_prompt_c="a woman face"
sprompt="in illustration style of Milton Glaser animals"
python launch.py --config configs/styletex.yaml --train --gradio --gpu 0 system.prompt_processor.prompt="a Leather Recliner ${sprompt}" system.guidance.ref_img_path=${image_path} system.guidance.ref_content_prompt="${image_prompt_c}" system.geometry.shape_init=mesh:load/shapes/sofa.obj trainer.max_steps=2500 system.geometry.shape_init_params=1.0

python launch.py --config configs/styletex.yaml --train --gradio --gpu 0 system.prompt_processor.prompt="a wooden treasure chest with metal accents and locks ${sprompt}" system.guidance.ref_img_path=${image_path} system.guidance.ref_content_prompt="${image_prompt_c}" system.geometry.shape_init=mesh:load/shapes/treasurechest.obj trainer.max_steps=2500 system.geometry.shape_init_params=0.8


image_path="load/style_images/elephant.jpeg"
image_prompt_c="an elephant"
sprompt="in colorful drawing style"
python launch.py --config configs/styletex.yaml --train --gradio --gpu 0 system.prompt_processor.prompt="a stool ${sprompt}" system.guidance.ref_img_path=${image_path} system.guidance.ref_content_prompt="${image_prompt_c}" system.geometry.shape_init=mesh:load/shapes/stool.obj trainer.max_steps=2500 system.geometry.shape_init_params=0.8

python launch.py --config configs/styletex.yaml --train --gradio --gpu 0 system.prompt_processor.prompt="a cupcake ${sprompt}" system.guidance.ref_img_path=${image_path} system.guidance.ref_content_prompt="${image_prompt_c}" system.geometry.shape_init=mesh:load/shapes/cupcake.obj trainer.max_steps=2500 system.geometry.shape_init_params=0.7


image_path="load/style_images/wooden_car.png"
image_prompt_c="a car"
sprompt="in a meticulously crafted wooden style"
python launch.py --config configs/styletex.yaml --train --gradio --gpu 0 system.prompt_processor.prompt="a seashell ${sprompt}" system.guidance.ref_img_path=${image_path} system.guidance.ref_content_prompt="${image_prompt_c}" system.geometry.shape_init=mesh:load/shapes/seashell.obj trainer.max_steps=2500 system.geometry.shape_init_params=1.0

python launch.py --config configs/styletex.yaml --train --gradio --gpu 0 system.prompt_processor.prompt="a nike shoe ${sprompt}" system.guidance.ref_img_path=${image_path} system.guidance.ref_content_prompt="${image_prompt_c}" system.geometry.shape_init=mesh:load/shapes/nike.obj trainer.max_steps=2500 system.geometry.shape_init_params=1.0


image_path="load/style_images/durer_dog.jpeg"
image_prompt_c="a dog"
sprompt="in Durer's sketch style"
python launch.py --config configs/styletex.yaml --train --gradio --gpu 0 system.prompt_processor.prompt="a strawberry ${sprompt}" system.guidance.ref_img_path=${image_path} system.guidance.ref_content_prompt="${image_prompt_c}" system.geometry.shape_init=mesh:load/shapes/strawberry.obj trainer.max_steps=2500 system.geometry.shape_init_params=0.8

python launch.py --config configs/styletex.yaml --train --gradio --gpu 0 system.prompt_processor.prompt="a teapot ${sprompt}" system.guidance.ref_img_path=${image_path} system.guidance.ref_content_prompt="${image_prompt_c}" system.geometry.shape_init=mesh:load/shapes/teapot.obj trainer.max_steps=2500 system.geometry.shape_init_params=0.8

