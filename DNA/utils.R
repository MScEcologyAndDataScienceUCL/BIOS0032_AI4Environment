count_sequences_in_file <- function(path, sequences) {
  reads <- ShortRead::sread(ShortRead::readFastq(path))

  # For each provided sequence, count the number of times it appears in the
  # reads
  lapply(sequences, function(sequence) {
    # Convert the sequence to a DNAString object and create a list of all
    # possible orientations (forward, reverse, and reverse complement)
    dna <- Biostrings::DNAString(sequence)
    orientations <- c(
      fwd = dna,
      fwd_comp = Biostrings::complement(dna),
      rev = Biostrings::reverse(dna),
      rev_comp = Biostrings::reverseComplement(dna)
    )

    # Count the number of reads in which the primer is found at least once
    # for every possible orientation
    counts <- lapply(orientations, function(seq) {
      sum(Biostrings::vcountPattern(seq, reads, fixed = FALSE) > 0)
    })

    # Add the total number of reads in which the primer is found regardless of
    # orientation
    counts |> purrr::list_assign(total = sum(unlist(counts)))
  })
}

count_primers <- function(df, primers, base_dir) {
  df |>
    dplyr::bind_cols(
      file.path(base_dir, df$file) |>
        purrr::map(
          \(path) count_sequences_in_file(path, primers)
        ) |>
        purrr::map_dfr(
          \(file_res) as.data.frame(t(unlist(file_res)))
        )
    )
}

run_cutadapt <- function(
  forward_path,
  rev_path,
  forward_output,
  rev_output,
  primers,
  count = 1,
  minimum_length = 10,
  cores = 0
) {
  system2(
    "uvx",
    args = c(
      "cutadapt",
      "--revcomp",
      "--match-read-wildcards",
      "--discard-untrimmed",
      "-g",
      primers["forward"],
      "-a",
      dada2::rc(primers["reverse"]),
      "-G",
      primers["reverse"],
      "-A",
      dada2::rc(primers["forward"]),
      "--times",
      count,
      "--minimum-length",
      minimum_length,
      "--cores",
      cores,
      "--output",
      forward_output,
      "--paired-output",
      rev_output,
      forward_path,
      rev_path
    ),
  )
}

remove_primers <- function(
  df,
  primers,
  base_dir,
  output_dir,
  count = 2,
  minimum_length = 10,
  cores = 0
) {
  df |>
    purrr::pmap(\(..., forward, reverse) {
      run_cutadapt(
        file.path(base_dir, forward),
        file.path(base_dir, reverse),
        file.path(output_dir, forward),
        file.path(output_dir, reverse),
        primers,
        count = count,
        minimum_length = minimum_length,
        cores = cores
      )
    })
}

read_taxa_from_basta <- function(path) {
  read.table(path, sep = "\t", col.names = c("ASV_ID", "Taxonomy")) |>
    tidyr::separate(
      "Taxonomy",
      into = c(
        "Domain",
        "Phylum",
        "Class",
        "Order",
        "Family",
        "Genus",
        "Species"
      ),
      sep = ";",
      fill = "right",
      extra = "drop"
    ) |>
    tibble::column_to_rownames("ASV_ID") |>
    dplyr::mutate_if(is.character, list(~ dplyr::na_if(., ""))) |>
    tidyr::replace_na(list(
      Species = "unknown",
      Genus = "unknown",
      Family = "unknown"
    )) |>
    as.matrix()
}
