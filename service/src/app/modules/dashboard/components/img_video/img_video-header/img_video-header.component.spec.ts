import { ComponentFixture, TestBed } from '@angular/core/testing';

import { ImgVideoHeaderComponent } from './img_video-header.component';

describe('ImgVideoHeaderComponent', () => {
  let component: ImgVideoHeaderComponent;
  let fixture: ComponentFixture<ImgVideoHeaderComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
    imports: [ImgVideoHeaderComponent],
}).compileComponents();
  });

  beforeEach(() => {
    fixture = TestBed.createComponent(ImgVideoHeaderComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
