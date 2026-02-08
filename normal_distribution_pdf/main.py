import math
def normal_pdf(x: float, mean: float, std_dev: float) -> float:
	"""
	Calculate the probability density function (PDF) of the normal distribution.
	:param x: The value at which the PDF is evaluated.
	:param mean: The mean (μ) of the distribution.
	:param std_dev: The standard deviation (σ) of the distribution.
	Formula: (1 / (std * sqrt(2*pi))) * e^(-0.5 * ((x-mean)/std)^2)
	"""

	if std_dev <= 0:
		raise ValueError("Standard deviation must be positive.")

	#1. Calculate the coefficient
	coeff = 1 / (std_dev * math.sqrt(2 * math.pi))

	#2. Calculate the exponent
	z_score = (x - mean) / std_dev
	exponent = -0.5 * (z_score ** 2)

	#3. Calculate the PDF value
	pdf_value = coeff * math.exp(exponent)

	return round(pdf_value, 5)

if __name__ == "__main__":
    print(f"Manual Test: {normal_pdf(16, 15, 2.04)}")   