package edu.stanford.bmir.protege.web.shared.place;

import com.google.gwt.place.shared.Place;

/**
 * Matthew Horridge
 * Stanford Center for Biomedical Informatics Research
 * 12/02/16
 */
public class ProjectListPlaceTokenizer implements WebProtegePlaceTokenizer<ProjectListPlace> {

    private static final String PROJECTS_LIST = "projects/list";

    @Override
    public boolean matches(String token) {
        return PROJECTS_LIST.equals(token);
    }

    public ProjectListPlace getPlace(String token) {
        return new ProjectListPlace();
    }

    public String getToken(ProjectListPlace place) {
        return PROJECTS_LIST;
    }

    @Override
    public boolean isTokenizerFor(Place place) {
        return place instanceof ProjectListPlace;
    }
}
