import matplotlib.pyplot as plt

def plot_cam(cams, arrow_len=1.0):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    for count, cam in enumerate(cams):
        try:
            camname = cam.image_name
        except:
            camname = str(count)
        # cam.world_view_transform is w2c^T
        view = cam.world_view_transform.detach().cpu()
        view_inv = torch.inverse(view)

        center = view_inv[3, :3].numpy()
        # Camera forward direction in world space.
        # If the arrow points backward, flip the sign.
        forward = view_inv[2, :3].numpy()

        ax.scatter(center[0], center[1], center[2], s=50, label = camname)
        ax.quiver(
            center[0], center[1], center[2],
            forward[0], forward[1], forward[2],
            length=arrow_len, normalize=True, color="b"
        )
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_title("Camera position and viewing direction")
    ax.view_init(elev = 0, azim=90)
    ax.legend()
    plt.tight_layout()
    fig.savefig('cam_loc_and_view.jpg')

allcams = tempScene.getAllCameras().copy()
allcams.append(custom_cam)