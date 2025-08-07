library(Hmisc)

# for categorical variables -- remap everything to 0/1 for the models in Python
relabel_binary.fn = function(x) {
  if(class(x) == "logical") {
    x <- factor(as.integer(samples$logical_col), levels = c(0, 1))#, labels = c("Yes", "No"))
  }else {
    x <- factor(x, levels = c(1,0))#, labels = c("Yes", "No"))
  }
  return(x)
}


prep_table1.fn = function(wide) {
  
  #these variables are TRUE/FALSE -- for table purposes, reassign as categorical
  wide = wide %>% mutate_at(c("fstrom1_baseline",
                              "bl_amphetamine",
                              "dudit4d_baseline",
                              "dudit4f_baseline",
                              "dudit4g_baseline",
                              "bl_prevtb",
                              "bl_art", 
                              "cannabis_use", 
                              "meth_use", 
                              "bl_mandrax",
                              # "bl_inh_monoresistant",
                              "inh_resistant",
                              "diabetes",
                              "cxr_cavity_chest_radiograph_1", # converted 2 (Unknown) to NA in Python already,
                              "cxr_miliary_chest_radiograph_1", # converted 2 (Unknown) to NA in Python already,
                              'cxr_lymph_chest_radiograph_1',
                              'cxr_effusion_chest_radiograph_1',
                              'cxr_granuloma_chest_radiograph_1',
                              # "covid",
                              "bilateral_infiltrates", # made this from the cxr_infiltrates_chest_radiograph_1 column
                              "smoked_substance_use"
  ), 
  relabel_binary.fn)
  
  
  ####FACTOR VARIABLES####
  
  #baseline demographics/clinical 
  wide$screen_sex <- factor(wide$screen_sex, levels=c(1,0))#, labels=c( "Male", "Female"))
  # wide$education_baseline <- factor(as.logical(wide$education_baseline), levels = c(T, F), labels = c("< grade 9", ">= grade 9"))
  # wide$bl_prison <- factor(wide$bl_prison, levels = c(1, 0), labels = c("Yes", "No"))
  # wide$prevtb_outcome <- factor(wide$prevtb_outcome, levels = c(1, 2, 3), labels = c("Cured", "Treatment Completed", "Treatment Defaulted"))
  wide$bmi <- factor(wide$bmi, levels = c( "Underweight", "Normal Weight", "Overweight and Obese"), ordered = TRUE)
  # wide$diabetes2 <- factor(wide$diabetes2, levels = c("Normal", "Pre-diabetes", "Diabetes"))
  # wide$mixed_ancestry_race <- factor(wide$mixed_ancestry_race, levels = c(1, 0), labels = c("Yes", "No"))
  # wide$screen_race <- factor(wide$screen_race, levels = c(1,2,3,4,5), labels = c("Mixed ancestry", "Black African", "White", "Indian/Asian", "Other"))
  wide$unemployment_baseline <- factor(as.logical(wide$unemployment_baseline), levels = c(T, F), labels=c("Unemployed", "Employed"))
  # wide$smoked_substance_use <- factor(as.logical(wide$smoked_substance_use), levels = c(T, F), labels = c( "Smoked Substance Use", "No Smoked Substance Use"))
  wide$age_cat <- factor(wide$age_cat, levels = c(1,2,3,4,5,6), labels = c("<20", "20-29", "30-39", "40-49", "50-59", "60+"))
  wide$smear_grade_baseline <- factor(wide$smear_grade_baseline, levels = c(0, 1, 2, 3, 4), ordered = TRUE)
  
  #medical hx
  wide$bl_medhx___5 <- factor(wide$bl_medhx___5, levels = c(1,0), labels = c("Yes","No"))
  wide$bl_meds___10 <- factor(wide$bl_meds___10, levels = c(1,0), labels = c("Yes","No"))
  wide$bl_meds___12 <- factor(wide$bl_meds___12, levels = c(1,0), labels = c("Yes","No"))
  
  #HIV
  wide$bl_hiv <- factor(wide$bl_hiv, levels=c(1,0))#,labels=c("Positive", "Negative"))
  wide$HIV_CD4 <- factor(wide$HIV_CD4, levels=c(0, 1, 2))#, labels=c("HIV-", "HIV+/high CD4", "HIV+/low CD4"))
  
  # wide$bl_artreg <- factor(wide$bl_artreg, levels = c(1, 2, 3), labels = c("Tenofovir 300mg/Emtricitabine 200mg/Efavirenz 600mg", "Lopinavir and Ritonavir 200mg; Lamivudine and Zidovudine 50mg", "Other"))
  # if (all(is.na(wide$hiv_new_diagnosis))) {
  #   # If all values are NA, assign NA as the factor
  #   wide$hiv_new_diagnosis <- factor(NA, levels = c(1, 0), labels = c("Diagnosis within 1 week of trt initiation", "Diagnosis > 1 week before trt initiation"))
  # } else {
  #   # Apply factor normally if there are non-NA values
  #   wide$hiv_new_diagnosis <- factor(wide$hiv_new_diagnosis, levels = c(1, 0), labels = c("Diagnosis within 1 week of trt initiation", "Diagnosis > 1 week before trt initiation"))
  # }
  
  #alcohol use
  # wide$peth_50 <- factor(wide$peth_50,levels = c(1, 0), labels = c("Yes", "No"))
  # wide$tlfb_highint_baseline <- factor(wide$tlfb_highint_baseline, levels = c(1,0), labels = c("Yes", "No"))
  # wide$tlfb_highint2_baseline <- factor(wide$tlfb_highint2_baseline, levels = c(1,2,3), labels = c("High intensity alcohol use", "Some alcohol use", "No alcohol use"))
  # wide$tlfb_heavyalc2_baseline <- factor(wide$tlfb_heavyalc2_baseline, levels = c(3,2,1), labels = c("No alcohol use", "Some alcohol use",  "Heavy alcohol use"))
  # wide$tlfb_heavyalc_baseline <- factor(wide$tlfb_heavyalc_baseline, levels = c(1,0), labels = c("Yes", "No"))
  # wide$problem_alcohol <- factor(as.logical(wide$problem_alcohol), levels = c(TRUE, FALSE), labels = c("Yes", "No"))
  # wide$screen_alc <- factor(wide$screen_alc, levels = c(1, 0), labels = c("Yes", "No"))
  # wide$peth_cat_baseline <- factor(wide$peth_cat_baseline, levels = c(1,2,3), labels = c("< 50ng/mL", "50-200 ng/mL", "> 200 ng/mL"))
  # wide$peth_pos_baseline <- factor(wide$peth_pos_baseline, levels = c(1, 0), labels = c("Yes", "No"))
  # wide$audit_cat_baseline <- factor(wide$audit_cat_baseline, levels = c(1,2,3), labels = c("Low risk/abstinence", "Moderate risk", "At risk/severe risk"))
  wide$alcohol_use <- factor(wide$alcohol_use, levels=c('low-risk', 'dependence', 'harmful to hazardous'), ordered = TRUE)
  
  #other bio-behaviorals
  # wide$household_hunger_bin_baseline <- factor(wide$household_hunger_bin_baseline, levels = c(2,1,0), labels = c("Moderate to severe", "Moderate to severe", "Little to none"))
  # wide$cesd_bin_baseline <- factor(wide$cesd_bin_baseline, levels = c(1,0), labels = c("High risk", "Low risk"))
  
  #microbiologics
  # wide$cxr_cavity_chest_radiograph_1 <- factor(wide$cxr_cavity_chest_radiograph_1, levels = c(1,0,2), c("Yes", "No", "Unknown"))
  wide$smear_pos_no_contam_sputum_specimen_1	<- factor(wide$smear_pos_no_contam_sputum_specimen_1, levels = c(1,0))#	, labels = c("AFB", "No AFB"))
  # wide$s_concafb_sputum_specimen_1 <- factor( wide$s_concafb_sputum_specimen_1, levels = c(0,4,1,2,3), labels = c("No AFB", "Scanty","+", "++", "+++"))
  # wide$smear_pos_TRUST_or_TB <-  factor(wide$smear_pos_TRUST_or_TB, levels = c(1,0), labels = c("Positive", "Negative"))
  
  # changed the encoding in Python
  # wide$smear_grade_baseline <- factor(wide$smear_grade_baseline, levels=c(0, 1, 2, 3, 4), ordered = TRUE) 
  
  return(wide)
}