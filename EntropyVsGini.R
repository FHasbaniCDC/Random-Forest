entropy <- function(p_v) {
  e <- 0
  for (p in p_v) {
    if (p == 0) {
      this_term <- 0
    } else {
      this_term <- -p * log2(p)
    }
    e <- e + this_term
  }
  return(e)
}
gini <- function(p_v) {
  e <- 0
  for (p in p_v) {
    if (p == 0) {
      this.term <- 0
    } else {
      this.term <- p * (1 - p)
    }
    e <- e + this.term
  }
  return(e)
}
The following R script draws Figure 76.

Gini index vs. entropyFigure 76: Gini index vs. entropy
entropy.v <- NULL
gini.v <- NULL
p.v <- seq(0, 1, by = 0.01)
for (p in p.v) {
  entropy.v <- c(entropy.v, (entropy(c(p, 1 - p))))
  gini.v <- c(gini.v, (gini(c(p, 1 - p))))
}
plot(p.v, gini.v, type = "l", ylim = c(0, 1),
     xlab = "percentage of class 1",col = "red",
     ylab = "impurity measure", cex.lab = 1.5,
     cex.axis = 1.5, cex.main = 1.5,cex.sub = 1.5)
lines(p.v, entropy.v, col = "blue")
legend("topleft", legend = c("Entropy", "Gini index"),
       col = c("blue", "red"), lty = c(1, 1), cex = 0.8)