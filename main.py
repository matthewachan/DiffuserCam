import diffuser_cam
import matplotlib.pyplot as plt

# Runs the reconstruction algorithm on real data.
def main():
    algo = diffuser_cam.Algorithm('config/algorithm.json')
    dataloader = diffuser_cam.DataLoader('config/real_data.json')
    psf, data = dataloader.load_real_data()
    algo.reconstruct(psf, data, show_im=True)

# Example script to evalute the algorithm reconstructions in simulation.
def run_sim():
    algo = diffuser_cam.Algorithm('config/algorithm.json')
    dataloader = diffuser_cam.DataLoader('config/sim_data.json')
    psf, data, gt = dataloader.load_sim_data()
    plt.figure()
    plt.title('Ground truth (downsampled)')
    plt.imshow(gt)
    algo.reconstruct(psf, data, show_im=True)

if __name__ == '__main__':
    # main()
    run_sim()