package edu.stanford.bmir.protege.web.shared.project;

import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.mockito.Mock;
import org.mockito.runners.MockitoJUnitRunner;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.not;
import static org.hamcrest.Matchers.startsWith;
import static org.hamcrest.core.Is.is;
import static org.hamcrest.core.IsEqual.equalTo;

/**
 * Matthew Horridge
 * Stanford Center for Biomedical Informatics Research
 * 21/02/15
 */
@RunWith(MockitoJUnitRunner.class)
public class CreateNewProjectResult_TestCase {


    private CreateNewProjectResult result;

    private CreateNewProjectResult otherResult;

    @Mock
    private ProjectDetails projectDetails;


    @Before
    public void setUp() throws Exception {
        result = new CreateNewProjectResult(projectDetails);
        otherResult = new CreateNewProjectResult(projectDetails);
    }

    @Test(expected = NullPointerException.class)
    public void shouldThrowNullPointerException() {
        new CreateNewProjectResult(null);
    }

    @Test
    public void shouldBeEqualToSelf() {
        assertThat(result, is(equalTo(result)));
    }

    @Test
    public void shouldNotBeEqualToNull() {
        assertThat(result, is(not(equalTo(null))));
    }

    @Test
    public void shouldBeEqualToOther() {
        assertThat(result, is(equalTo(otherResult)));
    }

    @Test
    public void shouldHaveSameHashCodeAsOther() {
        assertThat(result.hashCode(), is(otherResult.hashCode()));
    }

    @Test
    public void shouldGenerateToString() {
        assertThat(result.toString(), startsWith("CreateNewProjectResult"));
    }

    @Test
    public void shouldReturnSuppliedProjectDetails() {
        assertThat(result.getProjectDetails(), is(projectDetails));
    }
}