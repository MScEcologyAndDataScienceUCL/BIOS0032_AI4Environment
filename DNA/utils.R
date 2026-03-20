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


get_basta_bin <- function() {
  tool_dir <- system2(
    "uv",
    args = c("tool", "dir"),
    stdout = TRUE
  )

  paste0(tool_dir, "/basta/bin/basta")
}

create_blast_db <- function(fasta_path, output_dir) {
  if (!dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE)
  }

  system2(
    "makeblastdb",
    args = c(
      "-in",
      fasta_path,
      "-parse_seqids",
      "-dbtype",
      "nucl",
      "-out",
      file.path(output_dir, "blast_db")
    )
  )
}

run_blastn <- function(
  query_path,
  db_dir,
  output_path
) {
  db <- file.path(db_dir, "blast_db")

  system2(
    "blastn",
    args = c(
      "-query",
      query_path,
      "-db",
      db,
      "-outfmt",
      "'6 qseqid sseqid pident length mismatch gapopen qstart qend sstart send evalue bitscore'",
      "-out",
      output_path
    )
  )
}

create_basta_db <- function(mapping_path, name) {
  basta_bin <- get_basta_bin()

  system2(
    basta_bin,
    args = c(
      "create_db",
      mapping_path,
      paste0(name, "_mapping.db"),
      "0",
      "1"
    )
  )
}

run_basta <- function(
  output_dir,
  blast_path,
  database_name,
  identity = 98,
  alength = 200,
  percent = 80,
  minimum = 1
) {
  basta_bin <- get_basta_bin()

  if (!dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE)
  }

  system2(
    basta_bin,
    args = c(
      "sequence",
      "-i",
      identity,
      "-l",
      alength,
      "-p",
      percent,
      "-m",
      minimum,
      blast_path,
      file.path(output_dir, "basta_out.txt"),
      "-v",
      file.path(output_dir, "basta_vout.txt"),
      database_name
    )
  )
}
