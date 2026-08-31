import torch


class MultiResolutionSTFTLoss(torch.nn.Module):
    """Linear-spectrum loss with extra weight on articulation detail.

    It is used only while training and therefore has no inference-time cost.
    Two resolutions keep the overhead modest while covering short consonant
    transients and longer harmonic structure.
    """

    def __init__(
        self,
        resolutions=((512, 128, 512), (2048, 480, 2048)),
        high_frequency_emphasis=0.75,
    ):
        super().__init__()
        self.resolutions = tuple(resolutions)
        self.high_frequency_emphasis = float(high_frequency_emphasis)
        self._windows = {}

    def _magnitude(self, audio, n_fft, hop_length, win_length):
        audio = audio.float().squeeze(1)
        cache_key = (win_length, str(audio.device))
        if cache_key not in self._windows:
            self._windows[cache_key] = torch.hann_window(
                win_length, device=audio.device, dtype=torch.float32
            )
        spectrum = torch.stft(
            audio,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=self._windows[cache_key],
            return_complex=True,
        )
        return spectrum.abs().clamp_min(1e-5)

    def forward(self, real, generated):
        total = real.new_zeros((), dtype=torch.float32)
        for n_fft, hop_length, win_length in self.resolutions:
            real_mag = self._magnitude(real.detach(), n_fft, hop_length, win_length)
            generated_mag = self._magnitude(
                generated, n_fft, hop_length, win_length
            )
            frequency = torch.linspace(
                0.0, 1.0, real_mag.shape[1], device=real_mag.device
            ).view(1, -1, 1)
            weight = 1.0 + self.high_frequency_emphasis * frequency.square()
            difference = (real_mag - generated_mag) * weight
            spectral_convergence = torch.linalg.vector_norm(difference) / (
                torch.linalg.vector_norm(real_mag * weight) + 1e-6
            )
            log_magnitude = torch.mean(
                torch.abs(torch.log(real_mag) - torch.log(generated_mag)) * weight
            )
            total = total + spectral_convergence + log_magnitude
        return total / len(self.resolutions)


def feature_loss(fmap_r, fmap_g):
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            rl = rl.float().detach()
            gl = gl.float()
            loss += torch.mean(torch.abs(rl - gl))

    return loss * 2


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        dr = dr.float()
        dg = dg.float()
        r_loss = torch.mean((1 - dr) ** 2)
        g_loss = torch.mean(dg**2)
        loss += r_loss + g_loss
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())

    return loss, r_losses, g_losses


def generator_loss(disc_outputs):
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        dg = dg.float()
        l = torch.mean((1 - dg) ** 2)
        gen_losses.append(l)
        loss += l

    return loss, gen_losses


def kl_loss(z_p, logs_q, m_p, logs_p, z_mask):
    """
    z_p, logs_q: [b, h, t_t]
    m_p, logs_p: [b, h, t_t]
    """
    z_p = z_p.float()
    logs_q = logs_q.float()
    m_p = m_p.float()
    logs_p = logs_p.float()
    z_mask = z_mask.float()

    kl = logs_p - logs_q - 0.5
    kl += 0.5 * ((z_p - m_p) ** 2) * torch.exp(-2.0 * logs_p)
    kl = torch.sum(kl * z_mask)
    l = kl / torch.sum(z_mask)
    return l
