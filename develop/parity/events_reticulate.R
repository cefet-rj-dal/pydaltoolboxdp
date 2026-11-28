# Deep ++ event parity: R harbinger orchestration versus public Python classes.

pkgload::load_all("../daltoolboxdp", quiet = TRUE)
source("../harbinger/R/harbinger_utils_deviation_measures.R")
source("../harbinger/R/harbinger_utils_filter_criteria.R")
source("../harbinger/R/harbinger_utils_candidate_selection.R")
source("../harbinger/R/harbinger_utils.R")
source("../harbinger/R/harbinger.R")
source("../harbinger/R/han_autoencoder.R")
source("../harbinger/R/hcp_waypoint.R")

np <- reticulate::import("numpy", convert = FALSE)
torch <- reticulate::import("torch", convert = FALSE)
dp <- reticulate::import("pydaltoolboxdp", convert = FALSE)

reset_seed <- function(seed) {
  invisible(np$random$seed(as.integer(seed)))
  invisible(torch$manual_seed(as.integer(seed)))
}

serie <- sin(seq(0, 8 * pi, length.out = 130)) * 0.1
serie[71:130] <- serie[71:130] + 1.5
serie[104] <- serie[104] + 2

reset_seed(307L)
r_han <- han_autoencoder(
  6L, 3L, autoenc_ed,
  encoder_hidden_sizes = 8L, epochs = 2L, batch_size = 8L
)
r_han <- daltoolbox::fit(r_han, serie)
r_han_detection <- detect(r_han, serie)

reset_seed(307L)
py_han <- dp$HanAutoencoder(
  6L, 3L,
  encoder_hidden_sizes = 8L, epochs = 2L, batch_size = 8L
)
invisible(py_han$fit(serie))
py_han_detection <- py_han$detect(serie)
py_han_res <- as.numeric(reticulate::py_to_r(py_han_detection$attrs$get("res")))
py_han_event <- as.logical(reticulate::py_to_r(py_han_detection$get("event")$to_numpy()))

message(
  "Han residual max diff: ",
  max(abs(attr(r_han_detection, "res") - py_han_res), na.rm = TRUE),
  "; R events: ", paste(which(r_han_detection$event), collapse = ","),
  "; Python events: ", paste(which(py_han_event), collapse = ",")
)

stopifnot(
  isTRUE(all.equal(attr(r_han_detection, "res"), py_han_res, tolerance = 1e-7)),
  identical(as.logical(r_han_detection$event), py_han_event)
)

reset_seed(311L)
r_waypoint <- hcp_waypoint(
  6L, 3L,
  warmup = 40L, retrain_size = 20L, buffer_size = 25L,
  k_factor = 0.35, h_low = 0.5, h_high = 1,
  epochs_init = 2L, epochs_retrain = 2L,
  encoderclass = autoenc_ed,
  encoder_hidden_sizes = 8L, epochs = 2L, batch_size = 8L
)
r_waypoint <- daltoolbox::fit(r_waypoint, serie)
r_waypoint_detection <- detect(r_waypoint, serie)

reset_seed(311L)
py_waypoint <- dp$HcpWaypoint(
  6L, 3L,
  warmup = 40L, retrain_size = 20L, buffer_size = 25L,
  k_factor = 0.35, h_low = 0.5, h_high = 1,
  epochs_init = 2L, epochs_retrain = 2L,
  encoder_hidden_sizes = 8L, epochs = 2L, batch_size = 8L
)
invisible(py_waypoint$fit(serie))
py_waypoint_detection <- py_waypoint$detect(serie)
py_waypoint_res <- as.numeric(
  reticulate::py_to_r(py_waypoint_detection$attrs$get("res"))
)
py_waypoint_event <- as.logical(
  reticulate::py_to_r(py_waypoint_detection$get("event")$to_numpy())
)
py_waypoint_tau <- as.numeric(
  reticulate::py_to_r(py_waypoint_detection$attrs$get("tau"))
)

message(
  "Waypoint residual max diff: ",
  max(abs(attr(r_waypoint_detection, "res") - py_waypoint_res), na.rm = TRUE),
  "; R tau: ", paste(attr(r_waypoint_detection, "tau"), collapse = ","),
  "; Python tau: ", paste(py_waypoint_tau, collapse = ","),
  "; R events: ", paste(which(r_waypoint_detection$event), collapse = ","),
  "; Python events: ", paste(which(py_waypoint_event), collapse = ",")
)

stopifnot(
  isTRUE(all.equal(
    attr(r_waypoint_detection, "res"),
    py_waypoint_res,
    tolerance = 1e-7
  )),
  identical(
    as.logical(r_waypoint_detection$event),
    py_waypoint_event
  ),
  isTRUE(all.equal(
    as.numeric(attr(r_waypoint_detection, "tau")),
    py_waypoint_tau,
    tolerance = 1e-7
  ))
)

message("Deep ++ event parity: exact (tolerance=1e-7)")
