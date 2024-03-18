# Area of Research
---
## 1. Create computational tools to strengthen gene annotation and improve RNA seq visualizations. 

* This research area grew out of our study of stress-induced gene expression patterns in C. elegans. 
* We found that traditional GO term- based enrichment studies insufficiently described the data and were not amenable for graphical comparison of multiple data sets. 
* This project became a collaboration with Dr. Marian Walhout's lab in which we annotated C. elegans genes based on ___physiological___ or ___molecular functions___ or ___subcellular localization___. 
* We validated this tool on diverse publicly available RNA seq data sets and, in each case, were able to uncover enrichment of gene sets with likely biological relevance that was not apparent with GO searches. 
* We have hosted a web-based service (WormCat) which has been widely adopted by the C. elegans community, with thousands of uses and multiple citations in its first year. 
* We are continuing in the development of a WormCat 2.0 in which we are using advanced computational tools to classify genes of unknown function in C. elegans. 
* WormCat is becoming an important part of the C. elegans bioinformatic repertoire and this visualization tool could be adapted for use with any organism. 

## 2. Better understand genes with minimally known function. 
* We created a specialized category for these genes, referred to Unknown in WormCat 1.0 (Holdorf, et al. Genetics 2020) and Unassigned in WormCat 2.0(Higgins, et al. Genetics 2022). 
* In published tissue specific RNA seq studies, we noted that more than half the genes expressed came from categories such as ___Unassigned___ or those defined by a location or a single domain such as ___Transmembrane Domain___ or ___Transmembrane Transport___.  
* This aligns with data from the human genome that shows that most research focuses on a small number of genes. This biases interpretation of -omics studies, as _pathway analysis depends on annotations_, and reinforces study of well described pathways. 
* These “poorly annotated genes (PAGs)” may be important for robustness or redundant with well-described pathways or function outside the lab environment. 
* Many of the C. elegans PAGs have orthologs in other organisms and we reasoned that we could resolve phenotypes if the most appropriate assays were known. 
* Therefore, we are developing machine learning protocols to cluster the PAGs with genes of known function.
* These clusters will be classified with WormCat and directed to phenotyping strategies such as lipid accumulation, stress response/aging, neuronal function or developmental timing within our lab or with our collaborators. 
* Our goal is to produce a resource that can be used by the C. elegans community. 
* Identification of pathways promoting robustness is critical if we are going to understand the impact of particular mutations or therapeutics in broad populations. 
* These studies contradict the assumption that the most important genes and pathways have been identified and provide a mechanism to broaden our reach in understanding what our genes do.
