import { ComponentFixture, TestBed } from '@angular/core/testing';

import { DIHeaderComponent } from './di-header.component';

describe('DIHeaderComponent', () => {
  let component: DIHeaderComponent;
  let fixture: ComponentFixture<DIHeaderComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
    imports: [DIHeaderComponent],
}).compileComponents();
  });

  beforeEach(() => {
    fixture = TestBed.createComponent(DIHeaderComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
