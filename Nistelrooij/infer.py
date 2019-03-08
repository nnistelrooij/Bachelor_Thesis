import numpy as np
from scipy.stats import vonmises, norm
from sklearn.utils.extmath import cartesian


class PSI:
    def __init__(self, a_oto, b_oto, sigma_prior, kappa_ver, kappa_hor, tau, head, frames, rods):
        self.a_oto = a_oto
        self.b_oto = b_oto
        self.sigma_prior = sigma_prior
        self.kappa_ver = kappa_ver
        self.kappa_hor = kappa_hor
        self.tau = tau

        self.head = head
        self.frames = frames
        self.rods = rods

        self.n_head = np.size(self.head)
        self.n_frames = np.size(self.frames)
        self.n_rods = np.size(self.rods)

        self.dims = [len(self.a_oto), len(self.b_oto), len(self.sigma_prior), len(kappa_ver),
                     len(self.kappa_hor), len(self.tau), self.n_head, self.n_frames, self.n_rods]

        self.P_frame = np.zeros(self.dims)
        self.P_oto = np.zeros(self.dims)
        self.P_prior = np.zeros(self.dims)

    def prior(self):
        factor = np.prod(self.dims) / len(self.sigma_prior) / self.n_rods

        for i in range(len(self.sigma_prior)):
            P_prior = norm.pdf(self.rods, 0, self.sigma_prior[i])
            P_prior = P_prior / np.sum(P_prior)

            P_prior = np.tile(P_prior, factor)
            P_prior = P_prior.reshape(self.dims[:3] + self.dims[4:])
            self.P_prior[:, :, i, :, :, :, :, :] = P_prior

        self.P_prior = self.P_prior / (np.prod(self.dims) / self.n_rods)


    def oto(self):
        factor = np.prod(self.dims) / len(self.a_oto) / len(self.b_oto) / self.n_head / self.n_rods

        for i in range(len(self.a_oto)):
            for j in range(len(self.b_oto)):
                for k in range(self.n_head):
                    P_oto = norm.pdf(self.rods, self.head[k], self.a_oto[i] + self.b_oto[j] * self.head[k])
                    P_oto = P_oto / np.sum(P_oto)

                    P_oto = np.tile(P_oto, factor)
                    P_oto = P_oto.reshape(self.dims[2:6] + self.dims[7:])
                    self.P_oto[i, j, :, :, :, :, k, :] = P_oto

        self.P_oto = self.P_oto / (np.prod(self.dims) / self.n_rods)


    def frame(self):
        # Aocr is normally a free parameter (the uncompensated ocular counterroll)
        Aocr = 14.6 * np.pi / 180  # convert to radians and fixed across subjects

        factor = len(self.a_oto) * len(self.b_oto) * len(self.sigma_prior)

        for i in range(len(self.kappa_ver)):
            for j in range(len(self.kappa_hor)):
                for k in range(len(self.tau)):
                    for l in range(self.n_head):
                        for m in range(self.n_frames):
                            # the frame in retinal coordinates
                            frame_retinal = -(self.frames[m] - self.head[l]) - Aocr * np.sin(self.head[l])
                            # make sure we stay in the -45 to 45 deg range
                            if frame_retinal > np.pi / 4:
                                frame_retinal = frame_retinal - np.pi / 2
                            elif frame_retinal < -np.pi / 4:
                                frame_retinal = frame_retinal + np.pi / 2

                            # compute how the kappa's changes with frame angle
                            kappa1 = self.kappa_ver[i] - (1 - np.cos(np.abs(2 * frame_retinal))) * self.tau[k] * (
                                    self.kappa_ver[i] - self.kappa_hor[j])
                            kappa2 = self.kappa_hor[j] + (1 - np.cos(np.abs(2 * frame_retinal))) * (
                                        1 - self.tau[k]) * (self.kappa_ver[i] - self.kappa_hor[j])

                            # probability distributions for the four von-mises
                            P_frame1 = vonmises.pdf(-self.rods + frame_retinal, kappa1)
                            P_frame2 = vonmises.pdf(-self.rods + np.pi / 2 + frame_retinal, kappa2)
                            P_frame3 = vonmises.pdf(-self.rods + np.pi + frame_retinal, kappa1)
                            P_frame4 = vonmises.pdf(-self.rods + 3 * np.pi / 2 + frame_retinal, kappa2)

                            # add the probability distributions
                            P_frame = (P_frame1 + P_frame2 + P_frame3 + P_frame4)
                            P_frame = P_frame / np.sum(P_frame)  # normalize to one

                            P_frame = np.tile(P_frame, factor)
                            P_frame = P_frame.reshape(self.dims[:3] + self.dims[8:])
                            self.P_frame[:, :, :, i, j, k, l, m] = P_frame

        self.P_frame = self.P_frame / (np.prod(self.dims) / self.n_rods)
        self.P_frame[np.isnan(self.P_frame)] = 1e-307

    def log_likelihood(self):
        log_prior = np.log(self.P_prior)
        log_oto = np.log(self.P_oto)
        log_frame = np.log(self.P_frame)

        self.log_lik = log_prior + log_oto + log_frame

        i = 1


psi = PSI(np.linspace(1, 3, 10) * np.pi / 180,
          np.linspace(0, 0.2, 10),
          np.linspace(4, 10, 10) * np.pi / 180,
          np.linspace(31.36, 1059.79, 10),
          np.linspace(0.3, 3.6, 10),
          np.linspace(0.6, 1, 10),
          [0, 30 * np.pi / 180],
          np.linspace(-45, 45, 10) * np.pi / 180,
          np.linspace(-7, 7, 5) * np.pi / 180,)
psi.prior()
psi.oto()
psi.frame()

psi.log_likelihood()

i = 4
